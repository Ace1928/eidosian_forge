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
def CONSTANT(self, node):
    if isinstance(node.value, str) and self._in_annotation:
        fn = functools.partial(self.handleStringAnnotation, node.value, node, node.lineno, node.col_offset, messages.ForwardAnnotationSyntaxError)
        self.deferFunction(fn)