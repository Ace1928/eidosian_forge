import __future__
import builtins
import ast
import collections
import contextlib
import doctest
import functools
import os
import re
import string
import sys
import warnings
from pyflakes import messages
def TUPLE(self, node):
    if isinstance(node.ctx, ast.Store):
        has_starred = False
        star_loc = -1
        for i, n in enumerate(node.elts):
            if isinstance(n, ast.Starred):
                if has_starred:
                    self.report(messages.TwoStarredExpressions, node)
                    break
                has_starred = True
                star_loc = i
        if star_loc >= 1 << 8 or len(node.elts) - star_loc - 1 >= 1 << 24:
            self.report(messages.TooManyExpressionsInStarredAssignment, node)
    self.handleChildren(node)