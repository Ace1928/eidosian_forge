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
class Converter5to6:
    """Perform an in-place upgrade of format 5 to format 6"""

    def convert(self, branch):
        format = BzrBranchFormat6()
        new_branch = format.open(branch.controldir, _found=True)
        new_branch._write_last_revision_info(*branch.last_revision_info())
        with new_branch.lock_write():
            new_branch.set_parent(branch.get_parent())
            new_branch.set_bound_location(branch.get_bound_location())
            new_branch.set_push_location(branch.get_push_location())
        new_branch.tags._set_tag_dict({})
        new_branch._transport.put_bytes('format', format.as_string(), mode=new_branch.controldir._get_file_mode())
        new_branch._transport.delete('revision-history')
        with branch.lock_write():
            try:
                branch.set_parent(None)
            except _mod_transport.NoSuchFile:
                pass
            branch.set_bound_location(None)