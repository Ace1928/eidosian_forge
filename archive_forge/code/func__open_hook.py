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
def _open_hook(self, possible_transports=None):
    if self._ignore_fallbacks:
        return
    if possible_transports is None:
        possible_transports = [self.controldir.root_transport]
    try:
        url = self.get_stacked_on_url()
    except (errors.UnstackableRepositoryFormat, errors.NotStacked, UnstackableBranchFormat):
        pass
    else:
        for hook in Branch.hooks['transform_fallback_location']:
            url = hook(self, url)
            if url is None:
                hook_name = Branch.hooks.get_hook_name(hook)
                raise AssertionError("'transform_fallback_location' hook %s returned None, not a URL." % hook_name)
        self._activate_fallback_location(url, possible_transports=possible_transports)