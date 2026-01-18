import math
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt
def _escape_chars(self, text):
    text = text.replace('\\', '\\[u005C]').replace('.', '\\[char46]').replace("'", '\\[u0027]').replace('`', '\\[u0060]').replace('~', '\\[u007E]')
    copy = text
    for char in copy:
        if len(char) != len(char.encode()):
            uni = char.encode('unicode_escape').decode()[1:].replace('x', 'u00').upper()
            text = text.replace(char, '\\[u' + uni[1:] + ']')
    return text