import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def _escapeRegexRangeChars(s):
    for c in '\\^-]':
        s = s.replace(c, _bslash + c)
    s = s.replace('\n', '\\n')
    s = s.replace('\t', '\\t')
    return _ustr(s)