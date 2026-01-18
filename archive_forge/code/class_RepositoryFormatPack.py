import re
import sys
from typing import Type
from ..lazy_import import lazy_import
import contextlib
import time
from breezy import (
from breezy.bzr import (
from breezy.bzr.index import (
from .. import errors, lockable_files, lockdir
from .. import transport as _mod_transport
from ..bzr import btree_index, index
from ..decorators import only_raises
from ..lock import LogicalLockResult
from ..repository import RepositoryWriteLockResult, _LazyListJoin
from ..trace import mutter, note, warning
from .repository import MetaDirRepository, RepositoryFormatMetaDir
from .serializer import Serializer
from .vf_repository import (MetaDirVersionedFileRepository,
class RepositoryFormatPack(MetaDirVersionedFileRepositoryFormat):
    """Format logic for pack structured repositories.

    This repository format has:
     - a list of packs in pack-names
     - packs in packs/NAME.pack
     - indices in indices/NAME.{iix,six,tix,rix}
     - knit deltas in the packs, knit indices mapped to the indices.
     - thunk objects to support the knits programming API.
     - a format marker of its own
     - an optional 'shared-storage' flag
     - an optional 'no-working-trees' flag
     - a LockDir lock
    """
    repository_class: Type[PackRepository]
    _commit_builder_class: Type[VersionedFileCommitBuilder]
    _serializer: Serializer
    supports_ghosts: bool = True
    supports_external_lookups: bool = False
    supports_chks: bool = False
    index_builder_class: Type[index.GraphIndexBuilder]
    index_class: Type[object]
    _fetch_uses_deltas: bool = True
    fast_deltas: bool = False
    supports_funky_characters: bool = True
    revision_graph_can_have_wrong_parents: bool = True

    def initialize(self, a_controldir, shared=False):
        """Create a pack based repository.

        :param a_controldir: bzrdir to contain the new repository; must already
            be initialized.
        :param shared: If true the repository will be initialized as a shared
                       repository.
        """
        mutter('creating repository in %s.', a_controldir.transport.base)
        dirs = ['indices', 'obsolete_packs', 'packs', 'upload']
        builder = self.index_builder_class()
        files = [('pack-names', builder.finish())]
        utf8_files = [('format', self.get_format_string())]
        self._upload_blank_content(a_controldir, dirs, files, utf8_files, shared)
        repository = self.open(a_controldir=a_controldir, _found=True)
        self._run_post_repo_init_hooks(repository, a_controldir, shared)
        return repository

    def open(self, a_controldir, _found=False, _override_transport=None):
        """See RepositoryFormat.open().

        :param _override_transport: INTERNAL USE ONLY. Allows opening the
                                    repository at a slightly different url
                                    than normal. I.e. during 'upgrade'.
        """
        if not _found:
            format = RepositoryFormatMetaDir.find_format(a_controldir)
        if _override_transport is not None:
            repo_transport = _override_transport
        else:
            repo_transport = a_controldir.get_repository_transport(None)
        control_files = lockable_files.LockableFiles(repo_transport, 'lock', lockdir.LockDir)
        return self.repository_class(_format=self, a_controldir=a_controldir, control_files=control_files, _commit_builder_class=self._commit_builder_class, _serializer=self._serializer)