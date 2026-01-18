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
def ARGUMENTS(self, node):
    self.handleChildren(node, omit=('defaults', 'kw_defaults'))