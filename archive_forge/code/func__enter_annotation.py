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
@contextlib.contextmanager
def _enter_annotation(self, ann_type=AnnotationState.BARE):
    orig, self._in_annotation = (self._in_annotation, ann_type)
    try:
        yield
    finally:
        self._in_annotation = orig