from contextlib import ExitStack
import breezy.config
from . import debug, errors, trace, ui
from .branch import Branch
from .errors import BzrError, ConflictsInTree, StrictCommitFailed
from .i18n import gettext
from .osutils import (get_user_encoding, is_inside_any, minimum_path_selection,
from .trace import is_quiet, mutter, note
from .urlutils import unescape_for_display
def _process_pre_hooks(self, old_revno, new_revno):
    """Process any registered pre commit hooks."""
    self._set_progress_stage('Running pre_commit hooks')
    self._process_hooks('pre_commit', old_revno, new_revno)