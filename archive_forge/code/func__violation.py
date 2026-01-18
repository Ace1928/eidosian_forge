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
def _violation(self, operation, *args, **kw):
    from setuptools.sandbox import SandboxViolation
    raise SandboxViolation(operation, args, kw)