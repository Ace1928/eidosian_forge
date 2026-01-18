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
class TestCrackFortran(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'crackfortran', 'gh2848.f90')]

    def test_gh2848(self):
        r = self.module.gh2848(1, 2)
        assert r == (1, 2)