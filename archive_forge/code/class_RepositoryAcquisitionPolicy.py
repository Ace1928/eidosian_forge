from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
class RepositoryAcquisitionPolicy:
    """Abstract base class for repository acquisition policies.

    A repository acquisition policy decides how a ControlDir acquires a repository
    for a branch that is being created.  The most basic policy decision is
    whether to create a new repository or use an existing one.
    """

    def __init__(self, stack_on, stack_on_pwd, require_stacking):
        """Constructor.

        Args:
          stack_on: A location to stack on
          stack_on_pwd: If stack_on is relative, the location it is
            relative to.
          require_stacking: If True, it is a failure to not stack.
        """
        self._stack_on = stack_on
        self._stack_on_pwd = stack_on_pwd
        self._require_stacking = require_stacking

    def configure_branch(self, branch):
        """Apply any configuration data from this policy to the branch.

        Default implementation sets repository stacking.
        """
        if self._stack_on is None:
            return
        if self._stack_on_pwd is None:
            stack_on = self._stack_on
        else:
            try:
                stack_on = urlutils.rebase_url(self._stack_on, self._stack_on_pwd, branch.user_url)
            except urlutils.InvalidRebaseURLs:
                stack_on = self._get_full_stack_on()
        try:
            branch.set_stacked_on_url(stack_on)
        except (_mod_branch.UnstackableBranchFormat, errors.UnstackableRepositoryFormat):
            if self._require_stacking:
                raise

    def requires_stacking(self):
        """Return True if this policy requires stacking."""
        return self._stack_on is not None and self._require_stacking

    def _get_full_stack_on(self):
        """Get a fully-qualified URL for the stack_on location."""
        if self._stack_on is None:
            return None
        if self._stack_on_pwd is None:
            return self._stack_on
        else:
            return urlutils.join(self._stack_on_pwd, self._stack_on)

    def _add_fallback(self, repository, possible_transports=None):
        """Add a fallback to the supplied repository, if stacking is set."""
        stack_on = self._get_full_stack_on()
        if stack_on is None:
            return
        try:
            stacked_dir = ControlDir.open(stack_on, possible_transports=possible_transports)
        except errors.JailBreak:
            return
        try:
            stacked_repo = stacked_dir.open_branch().repository
        except errors.NotBranchError:
            stacked_repo = stacked_dir.open_repository()
        try:
            repository.add_fallback_repository(stacked_repo)
        except errors.UnstackableRepositoryFormat:
            if self._require_stacking:
                raise
        else:
            self._require_stacking = True

    def acquire_repository(self, make_working_trees=None, shared=False, possible_transports=None):
        """Acquire a repository for this controlrdir.

        Implementations may create a new repository or use a pre-exising
        repository.

        Args:
          make_working_trees: If creating a repository, set
            make_working_trees to this value (if non-None)
          shared: If creating a repository, make it shared if True
        Returns:
          A repository, is_new_flag (True if the repository was created).
        """
        raise NotImplementedError(RepositoryAcquisitionPolicy.acquire_repository)