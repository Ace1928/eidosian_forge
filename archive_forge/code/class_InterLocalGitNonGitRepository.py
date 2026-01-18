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
class InterLocalGitNonGitRepository(InterGitNonGitRepository):
    """InterRepository that copies revisions from a local Git into a non-Git
    repository."""

    def fetch_objects(self, determine_wants, mapping, limit=None, lossy=False):
        """See `InterGitNonGitRepository`."""
        self._warn_slow()
        remote_refs = self.source.controldir.get_refs_container().as_dict()
        wants = determine_wants(remote_refs)
        target_git_object_retriever = get_object_store(self.target, mapping)
        with ui.ui_factory.nested_progress_bar() as pb:
            target_git_object_retriever.lock_write()
            try:
                pack_hint, last_rev = import_git_objects(self.target, mapping, self.source._git.object_store, target_git_object_retriever, wants, pb, limit)
                return (pack_hint, last_rev, remote_refs)
            finally:
                target_git_object_retriever.unlock()

    @staticmethod
    def is_compatible(source, target):
        """Be compatible with GitRepository."""
        if not isinstance(source, LocalGitRepository):
            return False
        if not target.supports_rich_root():
            return False
        if isinstance(target, GitRepository):
            return False
        if not getattr(target._format, 'supports_full_versioned_files', True):
            return False
        return True