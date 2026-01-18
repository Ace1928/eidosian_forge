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
class DummyWrapper(CodeWrapper):
    """Class used for testing independent of backends """
    template = '# dummy module for testing of SymPy\ndef %(name)s():\n    return "%(expr)s"\n%(name)s.args = "%(args)s"\n%(name)s.returns = "%(retvals)s"\n'

    def _prepare_files(self, routine):
        return

    def _generate_code(self, routine, helpers):
        with open('%s.py' % self.module_name, 'w') as f:
            printed = ', '.join([str(res.expr) for res in routine.result_variables])
            args = filter(lambda x: not isinstance(x, OutputArgument), routine.arguments)
            retvals = []
            for val in routine.result_variables:
                if isinstance(val, Result):
                    retvals.append('nameless')
                else:
                    retvals.append(val.result_var)
            print(DummyWrapper.template % {'name': routine.name, 'expr': printed, 'args': ', '.join([str(a.name) for a in args]), 'retvals': ', '.join([str(val) for val in retvals])}, end='', file=f)

    def _process_files(self, routine):
        return

    @classmethod
    def _get_wrapped_function(cls, mod, name):
        return getattr(mod, name)