import datetime
import time
import re
import numbers
import functools
import contextlib
from numbers import Number
from typing import Union, Tuple, Iterable
from typing import cast
def _check_unmatched(matches, text):
    """
    Ensure no words appear in unmatched text.
    """

    def check_unmatched(unmatched):
        found = re.search('\\w+', unmatched)
        if found:
            raise ValueError(f'Unexpected {found.group(0)!r}')
    pos = 0
    for match in matches:
        check_unmatched(text[pos:match.start()])
        yield match
        pos = match.end()
    check_unmatched(text[pos:])