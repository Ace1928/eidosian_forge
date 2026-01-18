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
class BzrBranch6(BzrBranch7):
    """See BzrBranchFormat6 for the capabilities of this branch.

    This subclass of BzrBranch7 disables the new features BzrBranch7 added,
    i.e. stacking.
    """

    def get_stacked_on_url(self):
        raise UnstackableBranchFormat(self._format, self.user_url)