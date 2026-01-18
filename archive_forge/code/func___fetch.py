import operator
from .. import errors, ui
from ..i18n import gettext
from ..revision import NULL_REVISION
from ..trace import mutter
def __fetch(self):
    """Primary worker function.

        This initialises all the needed variables, and then fetches the
        requested revisions, finally clearing the progress bar.
        """
    self.count_total = 0
    self.file_ids_names = {}
    with ui.ui_factory.nested_progress_bar() as pb:
        pb.show_pct = pb.show_count = False
        pb.update(gettext('Finding revisions'), 0, 2)
        search_result = self._revids_to_fetch()
        mutter('fetching: %s', str(search_result))
        if search_result.is_empty():
            return
        pb.update(gettext('Fetching revisions'), 1, 2)
        self._fetch_everything_for_search(search_result)