from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt
def _get_style(self, tokentype):
    if tokentype in self._stylecache:
        return self._stylecache[tokentype]
    otokentype = tokentype
    while not self.style.styles_token(tokentype):
        tokentype = tokentype.parent
    value = self.style.style_for_token(tokentype)
    result = ''
    if value['color']:
        result = ' fill="#' + value['color'] + '"'
    if value['bold']:
        result += ' font-weight="bold"'
    if value['italic']:
        result += ' font-style="italic"'
    self._stylecache[otokentype] = result
    return result