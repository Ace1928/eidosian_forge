import gzip
import os
from io import BytesIO
from ...lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from ... import debug, errors, lockable_files, lockdir, osutils, trace
from ... import transport as _mod_transport
from ... import urlutils
from ...bzr import tuned_gzip, versionedfile, weave, weavefile
from ...bzr.repository import RepositoryFormatMetaDir
from ...bzr.versionedfile import (AbsentContentFactory, FulltextContentFactory,
from ...bzr.vf_repository import (InterSameDataRepository,
from ...repository import InterRepository
from . import bzrdir as weave_bzrdir
from .store.text import TextStore
class PreSplitOutRepositoryFormat(VersionedFileRepositoryFormat):
    """Base class for the pre split out repository formats."""
    rich_root_data = False
    supports_tree_reference = False
    supports_ghosts = False
    supports_external_lookups = False
    supports_chks = False
    supports_nesting_repositories = True
    _fetch_order = 'topological'
    _fetch_reconcile = True
    fast_deltas = False
    supports_leaving_lock = False
    supports_overriding_transport = False
    revision_graph_can_have_wrong_parents = False

    def initialize(self, a_controldir, shared=False, _internal=False):
        """Create a weave repository."""
        if shared:
            raise errors.IncompatibleFormat(self, a_controldir._format)
        if not _internal:
            return self.open(a_controldir, _found=True)
        sio = BytesIO()
        weavefile.write_weave_v5(weave.Weave(), sio)
        empty_weave = sio.getvalue()
        trace.mutter('creating repository in %s.', a_controldir.transport.base)
        control_files = lockable_files.LockableFiles(a_controldir.transport, 'branch-lock', lockable_files.TransportLock)
        control_files.create_lock()
        control_files.lock_write()
        transport = a_controldir.transport
        try:
            transport.mkdir('revision-store', mode=a_controldir._get_dir_mode())
            transport.mkdir('weaves', mode=a_controldir._get_dir_mode())
            transport.put_bytes_non_atomic('inventory.weave', empty_weave, mode=a_controldir._get_file_mode())
        finally:
            control_files.unlock()
        repository = self.open(a_controldir, _found=True)
        self._run_post_repo_init_hooks(repository, a_controldir, shared)
        return repository

    def open(self, a_controldir, _found=False):
        """See RepositoryFormat.open()."""
        if not _found:
            raise NotImplementedError
        repo_transport = a_controldir.get_repository_transport(None)
        result = AllInOneRepository(_format=self, a_controldir=a_controldir)
        result.revisions = self._get_revisions(repo_transport, result)
        result.signatures = self._get_signatures(repo_transport, result)
        result.inventories = self._get_inventories(repo_transport, result)
        result.texts = self._get_texts(repo_transport, result)
        result.chk_bytes = None
        return result

    def is_deprecated(self):
        return True