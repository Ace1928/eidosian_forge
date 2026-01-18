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
def _partition_args(self, args):
    """Group function arguments into categories."""
    py_in = []
    py_out = []
    for arg in args:
        if isinstance(arg, OutputArgument):
            py_out.append(arg)
        elif isinstance(arg, InOutArgument):
            raise ValueError("Ufuncify doesn't support InOutArguments")
        else:
            py_in.append(arg)
    return (py_in, py_out)