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
def TYPEALIAS(self, node):
    self.handleNode(node.name, node)
    with self._type_param_scope(node):
        self.handle_annotation_always_deferred(node.value, node)