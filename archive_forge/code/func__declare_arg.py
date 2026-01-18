import sys
import os
import shutil
import tempfile
from subprocess import STDOUT, CalledProcessError, check_output
from string import Template
from warnings import warn
from sympy.core.cache import cacheit
from sympy.core.function import Lambda
from sympy.core.relational import Eq
from sympy.core.symbol import Dummy, Symbol
from sympy.tensor.indexed import Idx, IndexedBase
from sympy.utilities.codegen import (make_routine, get_code_generator,
from sympy.utilities.iterables import iterable
from sympy.utilities.lambdify import implemented_function
from sympy.utilities.decorator import doctest_depends_on
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
from setuptools.extension import Extension
from setuptools import setup
from numpy import get_include
def _declare_arg(self, arg):
    proto = self._prototype_arg(arg)
    if arg.dimensions:
        shape = '(' + ','.join((self._string_var(i[1] + 1) for i in arg.dimensions)) + ')'
        return proto + ' = np.empty({shape})'.format(shape=shape)
    else:
        return proto + ' = 0'