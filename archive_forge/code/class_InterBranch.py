from typing import (TYPE_CHECKING, Dict, List, Optional, TextIO, Tuple, Union,
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
import contextlib
import itertools
from . import config as _mod_config
from . import debug, errors, registry, repository
from . import revision as _mod_revision
from . import urlutils
from .controldir import (ControlComponent, ControlComponentFormat,
from .hooks import Hooks
from .inter import InterObject
from .lock import LogicalLockResult
from .revision import RevisionID
from .trace import is_quiet, mutter, mutter_callsite, note, warning
from .transport import Transport, get_transport
class InterBranch(InterObject[Branch]):
    """This class represents operations taking place between two branches.

    Its instances have methods like pull() and push() and contain
    references to the source and target repositories these operations
    can be carried out on.
    """
    _optimisers = []
    'The available optimised InterBranch types.'

    @classmethod
    def _get_branch_formats_to_test(klass):
        """Return an iterable of format tuples for testing.

        Returns: An iterable of (from_format, to_format) to use when testing
            this InterBranch class. Each InterBranch class should define this
            method itself.
        """
        raise NotImplementedError(klass._get_branch_formats_to_test)

    def pull(self, overwrite: bool=False, stop_revision: Optional[RevisionID]=None, possible_transports: Optional[List[Transport]]=None, local: bool=False, tag_selector=None) -> PullResult:
        """Mirror source into target branch.

        The target branch is considered to be 'local', having low latency.

        Returns: PullResult instance
        """
        raise NotImplementedError(self.pull)

    def push(self, overwrite: bool=False, stop_revision: Optional[RevisionID]=None, lossy: bool=False, _override_hook_source_branch: Optional[Branch]=None, tag_selector=None):
        """Mirror the source branch into the target branch.

        The source branch is considered to be 'local', having low latency.
        """
        raise NotImplementedError(self.push)

    def copy_content_into(self, revision_id=None, tag_selector=None):
        """Copy the content of source into target

        Args:
          revision_id:
            if not None, the revision history in the new branch will
            be truncated to end with revision_id.
          tag_selector: Optional callback that can decide
            to copy or not copy tags.
        """
        raise NotImplementedError(self.copy_content_into)

    def fetch(self, stop_revision: Optional[RevisionID]=None, limit: Optional[int]=None, lossy: bool=False) -> repository.FetchResult:
        """Fetch revisions.

        Args:
          stop_revision: Last revision to fetch
          limit: Optional rough limit of revisions to fetch
        Returns: FetchResult object
        """
        raise NotImplementedError(self.fetch)

    def update_references(self) -> None:
        """Import reference information from source to target.
        """
        raise NotImplementedError(self.update_references)

    @classmethod
    def get(self, source: Branch, target: Branch) -> 'InterBranch':
        return cast(InterBranch, super().get(source, target))