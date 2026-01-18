import re
def CUnescape(text: str) -> bytes:
    """Unescape a text string with C-style escape sequences to UTF-8 bytes.

  Args:
    text: The data to parse in a str.
  Returns:
    A byte string.
  """

    def ReplaceHex(m):
        if len(m.group(1)) & 1:
            return m.group(1) + 'x0' + m.group(2)
        return m.group(0)
    result = _CUNESCAPE_HEX.sub(ReplaceHex, text)
    result = result.encode('raw_unicode_escape').decode('raw_unicode_escape')
    result = result.encode('utf-8').decode('unicode_escape')
    return result.encode('latin-1')