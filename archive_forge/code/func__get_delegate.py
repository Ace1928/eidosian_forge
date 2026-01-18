from collections import namedtuple
from functools import singledispatch as simplegeneric
import importlib
import importlib.util
import importlib.machinery
import os
import os.path
import sys
from types import ModuleType
import warnings
def _get_delegate(self):
    finder = ImpImporter(self.filename)
    spec = _get_spec(finder, '__init__')
    return spec.loader