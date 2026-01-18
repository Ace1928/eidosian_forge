from contextlib import ExitStack
import breezy.config
from . import debug, errors, trace, ui
from .branch import Branch
from .errors import BzrError, ConflictsInTree, StrictCommitFailed
from .i18n import gettext
from .osutils import (get_user_encoding, is_inside_any, minimum_path_selection,
from .trace import is_quiet, mutter, note
from .urlutils import unescape_for_display
class ReportCommitToLog(NullCommitReporter):

    def _note(self, format, *args):
        """Output a message.

        Subclasses may choose to override this method.
        """
        note(format, *args)

    def snapshot_change(self, change, path):
        if path == '' and change in (gettext('added'), gettext('modified')):
            return
        self._note('%s %s', change, path)

    def started(self, revno, rev_id, location):
        self._note(gettext('Committing to: %s'), unescape_for_display(location, 'utf-8'))

    def completed(self, revno, rev_id):
        if revno is not None:
            self._note(gettext('Committed revision %d.'), revno)
            mutter('Committed revid %s as revno %d.', rev_id, revno)
        else:
            self._note(gettext('Committed revid %s.'), rev_id)

    def deleted(self, path):
        self._note(gettext('deleted %s'), path)

    def missing(self, path):
        self._note(gettext('missing %s'), path)

    def renamed(self, change, old_path, new_path):
        self._note('%s %s => %s', change, old_path, new_path)

    def is_verbose(self):
        return True