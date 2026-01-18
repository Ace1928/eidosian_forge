from fontTools.misc.textTools import tostr
import re
def _uniToUnicode(component):
    """Helper for toUnicode() to handle "uniABCD" components."""
    match = _re_uni.match(component)
    if match is None:
        return None
    digits = match.group(1)
    if len(digits) % 4 != 0:
        return None
    chars = [int(digits[i:i + 4], 16) for i in range(0, len(digits), 4)]
    if any((c >= 55296 and c <= 57343 for c in chars)):
        return None
    return ''.join([chr(c) for c in chars])