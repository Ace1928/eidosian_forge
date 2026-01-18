import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
def _escape_sub_callback(match):
    s = match.group()
    if len(s) == 2:
        try:
            return _ESCAPE_SUB_MAP[s]
        except KeyError:
            raise ValueError('Unsupported escape sequence: %s' % s)
    if s[1] == 'u':
        return chr(int(s[2:], 16))
    else:
        return chr(int(s[1:], 8))