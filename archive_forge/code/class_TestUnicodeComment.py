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
class TestUnicodeComment(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'crackfortran', 'unicode_comment.f90')]

    @pytest.mark.skipif(importlib.util.find_spec('charset_normalizer') is None, reason='test requires charset_normalizer which is not installed')
    def test_encoding_comment(self):
        self.module.foo(3)