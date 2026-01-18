from __future__ import absolute_import
import types
from . import Errors
def CodeRanges(code_list):
    """
    Given a list of codes as returned by chars_to_ranges, return
    an RE which will match a character in any of the ranges.
    """
    re_list = [CodeRange(code_list[i], code_list[i + 1]) for i in range(0, len(code_list), 2)]
    return Alt(*re_list)