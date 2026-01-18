from io import BytesIO
from typing import TYPE_CHECKING, Optional, Union
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from .. import errors, lockable_files
from .. import revision as _mod_revision
from .. import transport as _mod_transport
from .. import urlutils
from ..branch import (Branch, BranchFormat, BranchWriteLockResult,
from ..controldir import ControlDir
from ..decorators import only_raises
from ..lock import LogicalLockResult, _RelockDebugMixin
from ..trace import mutter
from . import bzrdir, rio
from .repository import MetaDirRepository
def _make_reference_clone_function(format, a_branch):
    """Create a clone() routine for a branch dynamically."""

    def clone(to_bzrdir, revision_id=None, repository_policy=None, name=None, tag_selector=None):
        """See Branch.clone()."""
        return format.initialize(to_bzrdir, target_branch=a_branch, name=name)
    return clone