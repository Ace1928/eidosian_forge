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
class InterLocalGitLocalGitRepository(InterGitGitRepository):
    source: LocalGitRepository
    target: LocalGitRepository

    def fetch_objects(self, determine_wants, limit=None, mapping=None, lossy: bool=False):
        if limit is not None:
            raise FetchLimitUnsupported(self)
        if lossy:
            raise LossyPushToSameVCS(self.source, self.target)
        from .remote import DefaultProgressReporter
        with ui.ui_factory.nested_progress_bar() as pb:
            progress = DefaultProgressReporter(pb).progress
            refs = self.source._git.fetch(self.target._git, determine_wants, progress=progress)
        return (None, None, refs)

    @staticmethod
    def is_compatible(source, target):
        """Be compatible with GitRepository."""
        return isinstance(source, LocalGitRepository) and isinstance(target, LocalGitRepository)