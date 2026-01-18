import re
import sys
from webencodings import ascii_lower
from .ast import (
from .serializer import serialize_string_value, serialize_url
def _consume_unicode_range(css, pos):
    """Return (range, new_pos)

    The given pos is assume to be just after the '+' of 'U+' or 'u+'.

    """
    length = len(css)
    start_pos = pos
    max_pos = min(pos + 6, length)
    while pos < max_pos and css[pos] in '0123456789abcdefABCDEF':
        pos += 1
    start = css[start_pos:pos]
    start_pos = pos
    while pos < max_pos and css[pos] == '?':
        pos += 1
    question_marks = pos - start_pos
    if question_marks:
        end = start + 'F' * question_marks
        start = start + '0' * question_marks
    elif pos + 1 < length and css[pos] == '-' and (css[pos + 1] in '0123456789abcdefABCDEF'):
        pos += 1
        start_pos = pos
        max_pos = min(pos + 6, length)
        while pos < max_pos and css[pos] in '0123456789abcdefABCDEF':
            pos += 1
        end = css[start_pos:pos]
    else:
        end = start
    return (int(start, 16), int(end, 16), pos)