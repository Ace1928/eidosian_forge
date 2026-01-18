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
class TestF77CommonBlockReader:

    def test_gh22648(self, tmp_path):
        fpath = util.getpath('tests', 'src', 'crackfortran', 'gh22648.pyf')
        with contextlib.redirect_stdout(io.StringIO()) as stdout_f2py:
            mod = crackfortran.crackfortran([str(fpath)])
        assert 'Mismatch' not in stdout_f2py.getvalue()