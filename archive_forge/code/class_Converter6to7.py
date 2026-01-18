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
class Converter6to7:
    """Perform an in-place upgrade of format 6 to format 7"""

    def convert(self, branch):
        format = BzrBranchFormat7()
        branch._set_config_location('stacked_on_location', '')
        branch._transport.put_bytes('format', format.as_string())