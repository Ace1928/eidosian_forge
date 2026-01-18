import os
import operator
import sys
import contextlib
import itertools
import unittest
from distutils.errors import DistutilsError, DistutilsOptionError
from distutils import log
from unittest import TestLoader
from pkg_resources import (
from .._importlib import metadata
from setuptools import Command
from setuptools.extern.more_itertools import unique_everseen
from setuptools.extern.jaraco.functools import pass_none
@staticmethod
@pass_none
def _resolve_as_ep(val):
    """
        Load the indicated attribute value, called, as a as if it were
        specified as an entry point.
        """
    return metadata.EntryPoint(value=val, name=None, group=None).load()()