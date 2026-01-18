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
class TestEval(util.F2PyTest):

    def test_eval_scalar(self):
        eval_scalar = crackfortran._eval_scalar
        assert eval_scalar('123', {}) == '123'
        assert eval_scalar('12 + 3', {}) == '15'
        assert eval_scalar('a + b', dict(a=1, b=2)) == '3'
        assert eval_scalar('"123"', {}) == "'123'"