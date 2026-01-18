import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _insert_thousands_sep(digits, spec, min_width=1):
    """Insert thousands separators into a digit string.

    spec is a dictionary whose keys should include 'thousands_sep' and
    'grouping'; typically it's the result of parsing the format
    specifier using _parse_format_specifier.

    The min_width keyword argument gives the minimum length of the
    result, which will be padded on the left with zeros if necessary.

    If necessary, the zero padding adds an extra '0' on the left to
    avoid a leading thousands separator.  For example, inserting
    commas every three digits in '123456', with min_width=8, gives
    '0,123,456', even though that has length 9.

    """
    sep = spec['thousands_sep']
    grouping = spec['grouping']
    groups = []
    for l in _group_lengths(grouping):
        if l <= 0:
            raise ValueError('group length should be positive')
        l = min(max(len(digits), min_width, 1), l)
        groups.append('0' * (l - len(digits)) + digits[-l:])
        digits = digits[:-l]
        min_width -= l
        if not digits and min_width <= 0:
            break
        min_width -= len(sep)
    else:
        l = max(len(digits), min_width, 1)
        groups.append('0' * (l - len(digits)) + digits[-l:])
    return sep.join(reversed(groups))