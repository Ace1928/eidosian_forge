import textwrap
from unicodedata import east_asian_width as _eawidth
from . import osutils
def _unicode_char_width(self, uc):
    """Return width of character `uc`.

        :param:     uc      Single unicode character.
        """
    return _eawidth(uc) in self._east_asian_doublewidth and 2 or 1