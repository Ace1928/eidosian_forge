import os
import os.path
import re
from debian.deprecation import function_deprecated_by
import debian._arch_table
@classmethod
def _order(cls, x):
    """Return an integer value for character x"""
    if x == '~':
        return -1
    if cls.re_digit.match(x):
        return int(x) + 1
    if cls.re_alpha.match(x):
        return ord(x)
    return ord(x) + 256