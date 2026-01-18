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
def _write_last_revision_info(self, revno, revision_id):
    """Simply write out the revision id, with no checks.

        Use set_last_revision_info to perform this safely.

        Does not update the revision_history cache.
        """
    out_string = b'%d %s\n' % (revno, revision_id)
    self._transport.put_bytes('last-revision', out_string, mode=self.controldir._get_file_mode())