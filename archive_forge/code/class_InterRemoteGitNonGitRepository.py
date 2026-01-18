import itertools
from typing import Callable, Dict, Tuple, Optional
from dulwich.errors import NotCommitError
from dulwich.objects import ObjectID
from dulwich.object_store import ObjectStoreGraphWalker
from dulwich.pack import PACK_SPOOL_FILE_MAX_SIZE
from dulwich.protocol import CAPABILITY_THIN_PACK, ZERO_SHA
from dulwich.refs import SYMREF
from dulwich.walk import Walker
from .. import config, trace, ui
from ..errors import (DivergedBranches, FetchLimitUnsupported,
from ..repository import FetchResult, InterRepository, AbstractSearchResult
from ..revision import NULL_REVISION, RevisionID
from .errors import NoPushSupport
from .fetch import DetermineWantsRecorder, import_git_objects
from .mapping import needs_roundtripping
from .object_store import get_object_store
from .push import MissingObjectsIterator, remote_divergence
from .refs import is_tag, ref_to_tag_name
from .remote import RemoteGitError, RemoteGitRepository
from .repository import GitRepository, GitRepositoryFormat, LocalGitRepository
from .unpeel_map import UnpeelMap
class InterRemoteGitNonGitRepository(InterGitNonGitRepository):
    """InterRepository that copies revisions from a remote Git into a non-Git
    repository."""

    def get_target_heads(self):
        all_revs = self.target.all_revision_ids()
        parent_map = self.target.get_parent_map(all_revs)
        all_parents = set()
        for values in parent_map.values():
            all_parents.update(values)
        return set(all_revs) - all_parents

    def fetch_objects(self, determine_wants, mapping, limit=None, lossy=False):
        """See `InterGitNonGitRepository`."""
        self._warn_slow()
        store = get_object_store(self.target, mapping)
        with store.lock_write():
            heads = self.get_target_heads()
            graph_walker = ObjectStoreGraphWalker([store._lookup_revision_sha1(head) for head in heads], lambda sha: store[sha].parents)
            wants_recorder = DetermineWantsRecorder(determine_wants)
            with ui.ui_factory.nested_progress_bar() as pb:
                objects_iter = self.source.fetch_objects(wants_recorder, graph_walker, store.get_raw)
                trace.mutter('Importing %d new revisions', len(wants_recorder.wants))
                pack_hint, last_rev = import_git_objects(self.target, mapping, objects_iter, store, wants_recorder.wants, pb, limit)
                return (pack_hint, last_rev, wants_recorder.remote_refs)

    @staticmethod
    def is_compatible(source, target):
        """Be compatible with GitRepository."""
        if not isinstance(source, RemoteGitRepository):
            return False
        if not target.supports_rich_root():
            return False
        if isinstance(target, GitRepository):
            return False
        if not getattr(target._format, 'supports_full_versioned_files', True):
            return False
        return True