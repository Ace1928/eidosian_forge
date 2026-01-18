from kivy.utils import platform
from kivy.core.clipboard import ClipboardBase
class ClipboardSDL2(ClipboardBase):

    def get(self, mimetype):
        return _get_text() if _has_text() else ''

    def _ensure_clipboard(self):
        super(ClipboardSDL2, self)._ensure_clipboard()
        self._encoding = 'utf8'

    def put(self, data=b'', mimetype='text/plain'):
        _set_text(data)

    def get_types(self):
        return ['text/plain']