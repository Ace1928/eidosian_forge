import contextlib
import sys
from typing import TYPE_CHECKING, Set, cast
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import config, controldir, errors, lockdir
from .. import transport as _mod_transport
from ..trace import mutter, note, warning
from ..transport import do_catching_redirections, local
def _find_source_repo(self, exit_stack, source_branch):
    """Find the source branch and repo for a sprout operation.

        This is helper intended for use by _sprout.

        :returns: (source_branch, source_repository).  Either or both may be
            None.  If not None, they will be read-locked (and their unlock(s)
            scheduled via the exit_stack param).
        """
    if source_branch is not None:
        exit_stack.enter_context(source_branch.lock_read())
        return (source_branch, source_branch.repository)
    try:
        source_branch = self.open_branch()
        source_repository = source_branch.repository
    except errors.NotBranchError:
        source_branch = None
        try:
            source_repository = self.open_repository()
        except errors.NoRepositoryPresent:
            source_repository = None
        else:
            exit_stack.enter_context(source_repository.lock_read())
    else:
        exit_stack.enter_context(source_branch.lock_read())
    return (source_branch, source_repository)