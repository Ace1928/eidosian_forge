import html
from urllib.parse import parse_qsl, quote, unquote, urlencode
from paste import request
def _html_attrs(self):
    attrs = list(self.attrs.items())
    onclick = 'window.open(%s); return false' % self._window_args()
    attrs.insert(0, ('target', self.params['target']))
    attrs.insert(0, ('onclick', onclick))
    attrs.insert(0, ('href', self.href))
    return attrs