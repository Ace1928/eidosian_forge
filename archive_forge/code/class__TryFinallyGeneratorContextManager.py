from __future__ import absolute_import
import cython
import os
import sys
import re
import io
import codecs
import glob
import shutil
import tempfile
from functools import wraps
from . import __version__ as cython_version
class _TryFinallyGeneratorContextManager(object):
    """
    Fast, bare minimum @contextmanager, only for try-finally, not for exception handling.
    """

    def __init__(self, gen):
        self._gen = gen

    def __enter__(self):
        return next(self._gen)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            next(self._gen)
        except (StopIteration, GeneratorExit):
            pass