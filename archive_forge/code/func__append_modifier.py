import sys
import encodings
import encodings.aliases
import re
import _collections_abc
from builtins import str as _builtin_str
import functools
def _append_modifier(code, modifier):
    if modifier == 'euro':
        if '.' not in code:
            return code + '.ISO8859-15'
        _, _, encoding = code.partition('.')
        if encoding in ('ISO8859-15', 'UTF-8'):
            return code
        if encoding == 'ISO8859-1':
            return _replace_encoding(code, 'ISO8859-15')
    return code + '@' + modifier