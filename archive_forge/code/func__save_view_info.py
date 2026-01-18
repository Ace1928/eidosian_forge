import re
from . import errors, osutils, transport
def _save_view_info(self):
    """Save the current view and all view definitions.

        Be sure to have initialised self._current and self._views before
        calling this method.
        """
    with self.tree.lock_write():
        if self._current is None:
            keywords = {}
        else:
            keywords = {'current': self._current}
        self.tree._transport.put_bytes('views', self._serialize_view_content(keywords, self._views))