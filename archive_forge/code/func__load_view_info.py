import re
from . import errors, osutils, transport
def _load_view_info(self):
    """Load the current view and dictionary of view definitions."""
    if not self._loaded:
        with self.tree.lock_read():
            try:
                view_content = self.tree._transport.get_bytes('views')
            except transport.NoSuchFile:
                self._current, self._views = (None, {})
            else:
                keywords, self._views = self._deserialize_view_content(view_content)
                self._current = keywords.get('current')
        self._loaded = True