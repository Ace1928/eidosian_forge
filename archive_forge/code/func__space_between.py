from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import collections
import contextlib
import itertools
import tokenize
from six import StringIO
from pasta.base import formatting as fmt
from pasta.base import fstring_utils
def _space_between(self, start_loc, end_loc):
    """Parse the space between a location and the next token"""
    if start_loc > end_loc:
        raise ValueError('start_loc > end_loc', start_loc, end_loc)
    if start_loc[0] > len(self.lines):
        return ''
    prev_row, prev_col = start_loc
    end_row, end_col = end_loc
    if prev_row == end_row:
        return self.lines[prev_row - 1][prev_col:end_col]
    return ''.join(itertools.chain((self.lines[prev_row - 1][prev_col:],), self.lines[prev_row:end_row - 1], (self.lines[end_row - 1][:end_col],) if end_col > 0 else ''))