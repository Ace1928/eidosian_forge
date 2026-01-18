import os
import pytest
import textwrap
import numpy as np
from . import util
class TestDocStringArguments(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'string', 'string.f')]

    def test_example(self):
        a = np.array(b'123\x00\x00')
        b = np.array(b'123\x00\x00')
        c = np.array(b'123')
        d = np.array(b'123')
        self.module.foo(a, b, c, d)
        assert a.tobytes() == b'123\x00\x00'
        assert b.tobytes() == b'B23\x00\x00'
        assert c.tobytes() == b'123'
        assert d.tobytes() == b'D23'