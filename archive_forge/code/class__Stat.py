import argparse
import pkgutil
import warnings
import types as pytypes
from numba.core import errors
from numba._version import get_versions
from numba.core.registry import cpu_target
from numba.tests.support import captured_stdout
class _Stat(object):
    """For gathering simple statistic of (un)supported functions"""

    def __init__(self):
        self.supported = 0
        self.unsupported = 0

    @property
    def total(self):
        total = self.supported + self.unsupported
        return total

    @property
    def ratio(self):
        ratio = self.supported / self.total * 100
        return ratio

    def describe(self):
        if self.total == 0:
            return 'empty'
        return 'supported = {supported} / {total} = {ratio:.2f}%'.format(supported=self.supported, total=self.total, ratio=self.ratio)

    def __repr__(self):
        return '{clsname}({describe})'.format(clsname=self.__class__.__name__, describe=self.describe())