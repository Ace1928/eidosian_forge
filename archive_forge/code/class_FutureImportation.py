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
class FutureImportation(ImportationFrom):
    """
    A binding created by a from `__future__` import statement.

    `__future__` imports are implicitly used.
    """

    def __init__(self, name, source, scope):
        super().__init__(name, source, '__future__')
        self.used = (scope, source)