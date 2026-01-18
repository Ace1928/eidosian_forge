from __future__ import annotations
import hashlib
import os
import sys
import typing as t
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
from hmac import HMAC
from pathlib import Path
from base64 import encodebytes
from jupyter_core.application import JupyterApp, base_flags
from traitlets import Any, Bool, Bytes, Callable, Enum, Instance, Integer, Unicode, default, observe
from traitlets.config import LoggingConfigurable, MultipleInstanceError
from . import NO_CONVERT, __version__, read, reads
class NotebookNotary(LoggingConfigurable):
    """A class for computing and verifying notebook signatures."""
    data_dir = Unicode(help='The storage directory for notary secret and database.').tag(config=True)

    @default('data_dir')
    def _data_dir_default(self):
        app = None
        try:
            if JupyterApp.initialized():
                app = JupyterApp.instance()
        except MultipleInstanceError:
            pass
        if app is None:
            app = JupyterApp()
            app.initialize(argv=[])
        return app.data_dir
    store_factory = Callable(help='A callable returning the storage backend for notebook signatures.\n         The default uses an SQLite database.').tag(config=True)

    @default('store_factory')
    def _store_factory_default(self):

        def factory():
            if sqlite3 is None:
                self.log.warning('Missing SQLite3, all notebooks will be untrusted!')
                return MemorySignatureStore()
            return SQLiteSignatureStore(self.db_file)
        return factory
    db_file = Unicode(help="The sqlite file in which to store notebook signatures.\n        By default, this will be in your Jupyter data directory.\n        You can set it to ':memory:' to disable sqlite writing to the filesystem.\n        ").tag(config=True)

    @default('db_file')
    def _db_file_default(self):
        if not self.data_dir:
            return ':memory:'
        return str(Path(self.data_dir) / 'nbsignatures.db')
    algorithm = Enum(algorithms, default_value='sha256', help='The hashing algorithm used to sign notebooks.').tag(config=True)

    @observe('algorithm')
    def _algorithm_changed(self, change):
        self.digestmod = getattr(hashlib, change['new'])
    digestmod = Any()

    @default('digestmod')
    def _digestmod_default(self):
        return getattr(hashlib, self.algorithm)
    secret_file = Unicode(help='The file where the secret key is stored.').tag(config=True)

    @default('secret_file')
    def _secret_file_default(self):
        if not self.data_dir:
            return ''
        return str(Path(self.data_dir) / 'notebook_secret')
    secret = Bytes(help='The secret key with which notebooks are signed.').tag(config=True)

    @default('secret')
    def _secret_default(self):
        if Path(self.secret_file).exists():
            with Path(self.secret_file).open('rb') as f:
                return f.read()
        else:
            secret = encodebytes(os.urandom(1024))
            self._write_secret_file(secret)
            return secret

    def __init__(self, **kwargs):
        """Initialize the notary."""
        super().__init__(**kwargs)
        self.store = self.store_factory()

    def _write_secret_file(self, secret):
        """write my secret to my secret_file"""
        self.log.info('Writing notebook-signing key to %s', self.secret_file)
        with Path(self.secret_file).open('wb') as f:
            f.write(secret)
        try:
            Path(self.secret_file).chmod(384)
        except OSError:
            self.log.warning('Could not set permissions on %s', self.secret_file)
        return secret

    def compute_signature(self, nb):
        """Compute a notebook's signature

        by hashing the entire contents of the notebook via HMAC digest.
        """
        hmac = HMAC(self.secret, digestmod=self.digestmod)
        with signature_removed(nb):
            for b in yield_everything(nb):
                hmac.update(b)
        return hmac.hexdigest()

    def check_signature(self, nb):
        """Check a notebook's stored signature

        If a signature is stored in the notebook's metadata,
        a new signature is computed and compared with the stored value.

        Returns True if the signature is found and matches, False otherwise.

        The following conditions must all be met for a notebook to be trusted:
        - a signature is stored in the form 'scheme:hexdigest'
        - the stored scheme matches the requested scheme
        - the requested scheme is available from hashlib
        - the computed hash from notebook_signature matches the stored hash
        """
        if nb.nbformat < 3:
            return False
        signature = self.compute_signature(nb)
        return self.store.check_signature(signature, self.algorithm)

    def sign(self, nb):
        """Sign a notebook, indicating that its output is trusted on this machine

        Stores hash algorithm and hmac digest in a local database of trusted notebooks.
        """
        if nb.nbformat < 3:
            return
        signature = self.compute_signature(nb)
        self.store.store_signature(signature, self.algorithm)

    def unsign(self, nb):
        """Ensure that a notebook is untrusted

        by removing its signature from the trusted database, if present.
        """
        signature = self.compute_signature(nb)
        self.store.remove_signature(signature, self.algorithm)

    def mark_cells(self, nb, trusted):
        """Mark cells as trusted if the notebook's signature can be verified

        Sets ``cell.metadata.trusted = True | False`` on all code cells,
        depending on the *trusted* parameter. This will typically be the return
        value from ``self.check_signature(nb)``.

        This function is the inverse of check_cells
        """
        if nb.nbformat < 3:
            return
        for cell in yield_code_cells(nb):
            cell['metadata']['trusted'] = trusted

    def _check_cell(self, cell, nbformat_version):
        """Do we trust an individual cell?

        Return True if:

        - cell is explicitly trusted
        - cell has no potentially unsafe rich output

        If a cell has no output, or only simple print statements,
        it will always be trusted.
        """
        if cell['metadata'].pop('trusted', False):
            return True
        if nbformat_version >= 4:
            unsafe_output_types = ['execute_result', 'display_data']
            safe_keys = {'output_type', 'execution_count', 'metadata'}
        else:
            unsafe_output_types = ['pyout', 'display_data']
            safe_keys = {'output_type', 'prompt_number', 'metadata'}
        for output in cell['outputs']:
            output_type = output['output_type']
            if output_type in unsafe_output_types:
                output_keys = set(output)
                if output_keys.difference(safe_keys):
                    return False
        return True

    def check_cells(self, nb):
        """Return whether all code cells are trusted.

        A cell is trusted if the 'trusted' field in its metadata is truthy, or
        if it has no potentially unsafe outputs.
        If there are no code cells, return True.

        This function is the inverse of mark_cells.
        """
        if nb.nbformat < 3:
            return False
        trusted = True
        for cell in yield_code_cells(nb):
            if not self._check_cell(cell, nb.nbformat):
                trusted = False
        return trusted