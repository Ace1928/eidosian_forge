from contextlib import ExitStack
import breezy.config
from . import debug, errors, trace, ui
from .branch import Branch
from .errors import BzrError, ConflictsInTree, StrictCommitFailed
from .i18n import gettext
from .osutils import (get_user_encoding, is_inside_any, minimum_path_selection,
from .trace import is_quiet, mutter, note
from .urlutils import unescape_for_display
def _check_pointless(self):
    if self.allow_pointless:
        return
    if len(self.parents) > 1:
        return
    if self.builder.any_changes():
        return
    raise PointlessCommit()