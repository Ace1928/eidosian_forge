import base64
import binascii
import re
import string
import six
def UnicodeEscape(s):
    """Replaces each non-ASCII character in s with an escape sequence.

  Non-ASCII characters are substituted with their 6-character unicode
  escape sequence \\uxxxx, where xxxx is a hex number.  The resulting
  string consists entirely of ASCII characters.  Existing escape
  sequences are unaffected, i.e., this operation is idempotent.

  Sample usage:
    >>> UnicodeEscape('asdf\\xff')
    'asdf\\\\u00ff'

  This escaping differs from the built-in s.encode('unicode_escape').  The
  built-in escape function uses hex escape sequences (e.g., '\\xe9') and escapes
  some control characters in lower ASCII (e.g., '\\x00').

  Args:
    s: string to be escaped

  Returns:
    escaped string
  """
    return _RE_NONASCII.sub(lambda m: '\\u%04x' % ord(m.group(0)), s)