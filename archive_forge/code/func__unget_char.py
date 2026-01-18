from io import StringIO
import sys
import dns.exception
import dns.name
import dns.ttl
from ._compat import long, text_type, binary_type
def _unget_char(self, c):
    """Unget a character.

        The unget buffer for characters is only one character large; it is
        an error to try to unget a character when the unget buffer is not
        empty.

        c: the character to unget
        raises UngetBufferFull: there is already an ungotten char
        """
    if self.ungotten_char is not None:
        raise UngetBufferFull
    self.ungotten_char = c