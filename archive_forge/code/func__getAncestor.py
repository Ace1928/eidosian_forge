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
def _getAncestor(self, node, ancestor_type):
    parent = node
    while True:
        if parent is self.root:
            return None
        parent = self.getParent(parent)
        if isinstance(parent, ancestor_type):
            return parent