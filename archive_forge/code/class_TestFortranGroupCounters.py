import importlib
import codecs
import time
import unicodedata
import pytest
import numpy as np
from numpy.f2py.crackfortran import markinnerspaces, nameargspattern
from . import util
from numpy.f2py import crackfortran
import textwrap
import contextlib
import io
class TestFortranGroupCounters(util.F2PyTest):

    def test_end_if_comment(self):
        fpath = util.getpath('tests', 'src', 'crackfortran', 'gh23533.f')
        try:
            crackfortran.crackfortran([str(fpath)])
        except Exception as exc:
            assert False, f"'crackfortran.crackfortran' raised an exception {exc}"