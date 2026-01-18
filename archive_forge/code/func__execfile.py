import os
import sys
import tempfile
import operator
import functools
import itertools
import re
import contextlib
import pickle
import textwrap
import builtins
import pkg_resources
from distutils.errors import DistutilsError
from pkg_resources import working_set
def _execfile(filename, globals, locals=None):
    """
    Python 3 implementation of execfile.
    """
    mode = 'rb'
    with open(filename, mode) as stream:
        script = stream.read()
    if locals is None:
        locals = globals
    code = compile(script, filename, 'exec')
    exec(code, globals, locals)