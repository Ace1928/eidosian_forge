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
class PackRepository(MetaDirVersionedFileRepository):
    """Repository with knit objects stored inside pack containers.

    The layering for a KnitPackRepository is:

    Graph        |  HPSS    | Repository public layer |
    ===================================================
    Tuple based apis below, string based, and key based apis above
    ---------------------------------------------------
    VersionedFiles
      Provides .texts, .revisions etc
      This adapts the N-tuple keys to physical knit records which only have a
      single string identifier (for historical reasons), which in older formats
      was always the revision_id, and in the mapped code for packs is always
      the last element of key tuples.
    ---------------------------------------------------
    GraphIndex
      A separate GraphIndex is used for each of the
      texts/inventories/revisions/signatures contained within each individual
      pack file. The GraphIndex layer works in N-tuples and is unaware of any
      semantic value.
    ===================================================

    """
    _commit_builder_class: Type[VersionedFileCommitBuilder]
    _serializer: Serializer

    def __init__(self, _format, a_controldir, control_files, _commit_builder_class, _serializer):
        MetaDirRepository.__init__(self, _format, a_controldir, control_files)
        self._commit_builder_class = _commit_builder_class
        self._serializer = _serializer
        self._reconcile_fixes_text_parents = True
        if self._format.supports_external_lookups:
            self._unstacked_provider = graph.CachingParentsProvider(self._make_parents_provider_unstacked())
        else:
            self._unstacked_provider = graph.CachingParentsProvider(self)
        self._unstacked_provider.disable_cache()

    def _all_revision_ids(self):
        """See Repository.all_revision_ids()."""
        with self.lock_read():
            return [key[0] for key in self.revisions.keys()]

    def _abort_write_group(self):
        self.revisions._index._key_dependencies.clear()
        self._pack_collection._abort_write_group()

    def _make_parents_provider(self):
        if not self._format.supports_external_lookups:
            return self._unstacked_provider
        return graph.StackedParentsProvider(_LazyListJoin([self._unstacked_provider], self._fallback_repositories))

    def _refresh_data(self):
        if not self.is_locked():
            return
        self._pack_collection.reload_pack_names()
        self._unstacked_provider.disable_cache()
        self._unstacked_provider.enable_cache()

    def _start_write_group(self):
        self._pack_collection._start_write_group()

    def _commit_write_group(self):
        hint = self._pack_collection._commit_write_group()
        self.revisions._index._key_dependencies.clear()
        self._unstacked_provider.disable_cache()
        self._unstacked_provider.enable_cache()
        return hint

    def suspend_write_group(self):
        tokens = self._pack_collection._suspend_write_group()
        self.revisions._index._key_dependencies.clear()
        self._write_group = None
        return tokens

    def _resume_write_group(self, tokens):
        self._start_write_group()
        try:
            self._pack_collection._resume_write_group(tokens)
        except errors.UnresumableWriteGroup:
            self._abort_write_group()
            raise
        for pack in self._pack_collection._resumed_packs:
            self.revisions._index.scan_unvalidated_index(pack.revision_index)

    def get_transaction(self):
        if self._write_lock_count:
            return self._transaction
        else:
            return self.control_files.get_transaction()

    def is_locked(self):
        return self._write_lock_count or self.control_files.is_locked()

    def is_write_locked(self):
        return self._write_lock_count

    def lock_write(self, token=None):
        """Lock the repository for writes.

        :return: A breezy.repository.RepositoryWriteLockResult.
        """
        locked = self.is_locked()
        if not self._write_lock_count and locked:
            raise errors.ReadOnlyError(self)
        self._write_lock_count += 1
        if self._write_lock_count == 1:
            self._transaction = transactions.WriteTransaction()
        if not locked:
            if 'relock' in debug.debug_flags and self._prev_lock == 'w':
                note('%r was write locked again', self)
            self._prev_lock = 'w'
            self._unstacked_provider.enable_cache()
            for repo in self._fallback_repositories:
                repo.lock_read()
            self._refresh_data()
        return RepositoryWriteLockResult(self.unlock, None)

    def lock_read(self):
        """Lock the repository for reads.

        :return: A breezy.lock.LogicalLockResult.
        """
        locked = self.is_locked()
        if self._write_lock_count:
            self._write_lock_count += 1
        else:
            self.control_files.lock_read()
        if not locked:
            if 'relock' in debug.debug_flags and self._prev_lock == 'r':
                note('%r was read locked again', self)
            self._prev_lock = 'r'
            self._unstacked_provider.enable_cache()
            for repo in self._fallback_repositories:
                repo.lock_read()
            self._refresh_data()
        return LogicalLockResult(self.unlock)

    def leave_lock_in_place(self):
        raise NotImplementedError(self.leave_lock_in_place)

    def dont_leave_lock_in_place(self):
        raise NotImplementedError(self.dont_leave_lock_in_place)

    def pack(self, hint=None, clean_obsolete_packs=False):
        """Compress the data within the repository.

        This will pack all the data to a single pack. In future it may
        recompress deltas or do other such expensive operations.
        """
        with self.lock_write():
            self._pack_collection.pack(hint=hint, clean_obsolete_packs=clean_obsolete_packs)

    def reconcile(self, other=None, thorough=False):
        """Reconcile this repository."""
        from .reconcile import PackReconciler
        with self.lock_write():
            reconciler = PackReconciler(self, thorough=thorough)
            return reconciler.reconcile()

    def _reconcile_pack(self, collection, packs, extension, revs, pb):
        raise NotImplementedError(self._reconcile_pack)

    @only_raises(errors.LockNotHeld, errors.LockBroken)
    def unlock(self):
        if self._write_lock_count == 1 and self._write_group is not None:
            self.abort_write_group()
            self._unstacked_provider.disable_cache()
            self._transaction = None
            self._write_lock_count = 0
            raise errors.BzrError('Must end write group before releasing write lock on %s' % self)
        if self._write_lock_count:
            self._write_lock_count -= 1
            if not self._write_lock_count:
                transaction = self._transaction
                self._transaction = None
                transaction.finish()
        else:
            self.control_files.unlock()
        if not self.is_locked():
            self._unstacked_provider.disable_cache()
            for repo in self._fallback_repositories:
                repo.unlock()