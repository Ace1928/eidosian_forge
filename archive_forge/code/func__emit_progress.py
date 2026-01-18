from contextlib import ExitStack
import breezy.config
from . import debug, errors, trace, ui
from .branch import Branch
from .errors import BzrError, ConflictsInTree, StrictCommitFailed
from .i18n import gettext
from .osutils import (get_user_encoding, is_inside_any, minimum_path_selection,
from .trace import is_quiet, mutter, note
from .urlutils import unescape_for_display
def _emit_progress(self):
    if self.pb_entries_count is not None:
        text = gettext('{0} [{1}] - Stage').format(self.pb_stage_name, self.pb_entries_count)
    else:
        text = gettext('%s - Stage') % (self.pb_stage_name,)
    self.pb.update(text, self.pb_stage_count, self.pb_stage_total)