import re
from email.errors import HeaderParseError
from email.parser import HeaderParser
from inspect import cleandoc
from django.urls import reverse
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import mark_safe
def _get_group_start_end(start, end, pattern):
    unmatched_open_brackets, prev_char = (1, None)
    for idx, val in enumerate(pattern[end:]):
        if val == '(' and prev_char != '\\':
            unmatched_open_brackets += 1
        elif val == ')' and prev_char != '\\':
            unmatched_open_brackets -= 1
        prev_char = val
        if unmatched_open_brackets == 0:
            return (start, end + idx + 1)