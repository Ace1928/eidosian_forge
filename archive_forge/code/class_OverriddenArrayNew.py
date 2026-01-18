import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
class OverriddenArrayNew(OverriddenArrayOld):

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        kwargs = kwargs.copy()
        if 'where' in kwargs:
            kwargs['where'] = self._unwrap((kwargs['where'],))
            if kwargs['where'] is NotImplemented:
                return NotImplemented
            else:
                kwargs['where'] = kwargs['where'][0]
        r = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        if r is not NotImplemented:
            r = r.view(type(self))
        return r