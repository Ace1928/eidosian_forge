import math
import textwrap
import sys
import pytest
import threading
import traceback
import time
import numpy as np
from numpy.testing import IS_PYPY
from . import util
class TestGH25211(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'callback', 'gh25211.f'), util.getpath('tests', 'src', 'callback', 'gh25211.pyf')]
    module_name = 'callback2'

    def test_gh18335(self):

        def bar(x):
            return x * x
        res = self.module.foo(bar)
        assert res == 110