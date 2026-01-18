from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
def _keep_original_ws(s, tag_s):
    """Replace whitespace with the original whitespace characters in `s`"""
    return ''.join((c if tag_c == ' ' and c.isspace() else tag_c for c, tag_c in zip(s, tag_s)))