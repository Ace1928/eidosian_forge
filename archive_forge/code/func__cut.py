import textwrap
from unicodedata import east_asian_width as _eawidth
from . import osutils
def _cut(self, s, width):
    """Returns head and rest of s. (head+rest == s)

        Head is large as long as _width(head) <= width.
        """
    w = 0
    charwidth = self._unicode_char_width
    for pos, c in enumerate(s):
        w += charwidth(c)
        if w > width:
            return (s[:pos], s[pos:])
    return (s, '')