import re
from numba.core import types
def _escape_string(text):
    """Escape the given string so that it only contains ASCII characters
    of [a-zA-Z0-9_$].

    The dollar symbol ($) and other invalid characters are escaped into
    the string sequence of "$xx" where "xx" is the hex codepoint of the char.

    Multibyte characters are encoded into utf8 and converted into the above
    hex format.
    """

    def repl(m):
        return ''.join(('_%02x' % ch for ch in m.group(0).encode('utf8')))
    ret = re.sub(_re_invalid_char, repl, text)
    if not isinstance(ret, str):
        return ret.encode('ascii')
    return ret