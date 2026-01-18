import errno
import os
from io import BytesIO
from .lazy_import import lazy_import
import gzip
import itertools
import patiencediff
from breezy import (
from . import errors
from .i18n import gettext
@staticmethod
def _from_patch(lines):
    """This is private because it is essential to split lines on 
 only"""
    line_iter = iter(lines)
    hunks = []
    cur_line = None
    while True:
        try:
            cur_line = next(line_iter)
        except StopIteration:
            break
        first_char = cur_line[0:1]
        if first_char == b'i':
            num_lines = int(cur_line.split(b' ')[1])
            hunk_lines = [next(line_iter) for _ in range(num_lines)]
            hunk_lines[-1] = hunk_lines[-1][:-1]
            hunks.append(NewText(hunk_lines))
        elif first_char == b'\n':
            hunks[-1].lines[-1] += b'\n'
        else:
            if not first_char == b'c':
                raise AssertionError(first_char)
            parent, parent_pos, child_pos, num_lines = (int(v) for v in cur_line.split(b' ')[1:])
            hunks.append(ParentText(parent, parent_pos, child_pos, num_lines))
    return MultiParent(hunks)