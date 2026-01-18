import re
from email.errors import HeaderParseError
from email.parser import HeaderParser
from inspect import cleandoc
from django.urls import reverse
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import mark_safe
def _find_groups(pattern, group_matcher):
    prev_end = None
    for match in group_matcher.finditer(pattern):
        if (indices := _get_group_start_end(match.start(0), match.end(0), pattern)):
            start, end = indices
            if prev_end and start > prev_end or not prev_end:
                yield (start, end, match)
            prev_end = end