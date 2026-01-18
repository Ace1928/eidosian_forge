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
def ANNASSIGN(self, node):
    self.handleAnnotation(node.annotation, node)
    if node.value:
        if _is_typing(node.annotation, 'TypeAlias', self.scopeStack):
            self.handleAnnotation(node.value, node)
        else:
            self.handleNode(node.value, node)
    self.handleNode(node.target, node)