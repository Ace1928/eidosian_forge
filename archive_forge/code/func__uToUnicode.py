from fontTools.misc.textTools import tostr
import re
def _uToUnicode(component):
    """Helper for toUnicode() to handle "u1ABCD" components."""
    match = _re_u.match(component)
    if match is None:
        return None
    digits = match.group(1)
    try:
        value = int(digits, 16)
    except ValueError:
        return None
    if value >= 0 and value <= 55295 or (value >= 57344 and value <= 1114111):
        return chr(value)
    return None