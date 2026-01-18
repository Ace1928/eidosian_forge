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
def _run_post_change_branch_tip_hooks(self, old_revno, old_revid):
    """Run the post_change_branch_tip hooks."""
    hooks = Branch.hooks['post_change_branch_tip']
    if not hooks:
        return
    new_revno, new_revid = self.last_revision_info()
    params = ChangeBranchTipParams(self, old_revno, new_revno, old_revid, new_revid)
    for hook in hooks:
        hook(params)