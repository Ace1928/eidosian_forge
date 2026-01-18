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
class TrustNotebookApp(JupyterApp):
    """An application for handling notebook trust."""
    version = __version__
    description = 'Sign one or more Jupyter notebooks with your key,\n    to trust their dynamic (HTML, Javascript) output.\n\n    Otherwise, you will have to re-execute the notebook to see output.\n    '

    @default('config_file_name')
    def _config_file_name_default(self):
        return 'jupyter_notebook_config'
    examples = '\n    jupyter trust mynotebook.ipynb and_this_one.ipynb\n    '
    flags = trust_flags
    reset = Bool(False, help='If True, delete the trusted signature cache.\n        After reset, all previously signed notebooks will become untrusted.\n        ').tag(config=True)
    notary = Instance(NotebookNotary)

    @default('notary')
    def _notary_default(self):
        return NotebookNotary(parent=self, data_dir=self.data_dir)

    def sign_notebook_file(self, notebook_path):
        """Sign a notebook from the filesystem"""
        if not Path(notebook_path).exists():
            self.log.error('Notebook missing: %s', notebook_path)
            self.exit(1)
        with Path(notebook_path).open(encoding='utf8') as f:
            nb = read(f, NO_CONVERT)
        self.sign_notebook(nb, notebook_path)

    def sign_notebook(self, nb, notebook_path='<stdin>'):
        """Sign a notebook that's been loaded"""
        if self.notary.check_signature(nb):
            print('Notebook already signed: %s' % notebook_path)
        else:
            print('Signing notebook: %s' % notebook_path)
            self.notary.sign(nb)

    def generate_new_key(self):
        """Generate a new notebook signature key"""
        print('Generating new notebook key: %s' % self.notary.secret_file)
        self.notary._write_secret_file(os.urandom(1024))

    def start(self):
        """Start the trust notebook app."""
        if self.reset:
            if Path(self.notary.db_file).exists():
                print('Removing trusted signature cache: %s' % self.notary.db_file)
                Path(self.notary.db_file).unlink()
            self.generate_new_key()
            return
        if not self.extra_args:
            self.log.debug('Reading notebook from stdin')
            nb_s = sys.stdin.read()
            assert isinstance(nb_s, str)
            nb = reads(nb_s, NO_CONVERT)
            self.sign_notebook(nb, '<stdin>')
        else:
            for notebook_path in self.extra_args:
                self.sign_notebook_file(notebook_path)