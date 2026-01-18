import os
import sys
import platform
import inspect
import traceback
import pdb
import re
import linecache
import time
from fnmatch import fnmatch
from timeit import default_timer as clock
import doctest as pdoctest  # avoid clashing with our doctest() function
from doctest import DocTestFinder, DocTestRunner
import random
import subprocess
import shutil
import signal
import stat
import tempfile
import warnings
from contextlib import contextmanager
from inspect import unwrap
from sympy.core.cache import clear_cache
from sympy.external import import_module
from sympy.external.gmpy import GROUND_TYPES, HAS_GMPY
from collections import namedtuple
def _get_doctest_blacklist():
    """Get the default blacklist for the doctests"""
    blacklist = []
    blacklist.extend(['doc/src/modules/plotting.rst', 'doc/src/modules/physics/mechanics/autolev_parser.rst', 'sympy/codegen/array_utils.py', 'sympy/core/compatibility.py', 'sympy/core/trace.py', 'sympy/galgebra.py', 'sympy/parsing/autolev/_antlr/autolevlexer.py', 'sympy/parsing/autolev/_antlr/autolevlistener.py', 'sympy/parsing/autolev/_antlr/autolevparser.py', 'sympy/parsing/latex/_antlr/latexlexer.py', 'sympy/parsing/latex/_antlr/latexparser.py', 'sympy/plotting/pygletplot/__init__.py', 'sympy/plotting/pygletplot/plot.py', 'sympy/printing/ccode.py', 'sympy/printing/cxxcode.py', 'sympy/printing/fcode.py', 'sympy/testing/randtest.py', 'sympy/this.py'])
    num = 12
    for i in range(1, num + 1):
        blacklist.append('sympy/parsing/autolev/test-examples/ruletest' + str(i) + '.py')
    blacklist.extend(['sympy/parsing/autolev/test-examples/pydy-example-repo/mass_spring_damper.py', 'sympy/parsing/autolev/test-examples/pydy-example-repo/chaos_pendulum.py', 'sympy/parsing/autolev/test-examples/pydy-example-repo/double_pendulum.py', 'sympy/parsing/autolev/test-examples/pydy-example-repo/non_min_pendulum.py'])
    if import_module('numpy') is None:
        blacklist.extend(['sympy/plotting/experimental_lambdify.py', 'sympy/plotting/plot_implicit.py', 'examples/advanced/autowrap_integrators.py', 'examples/advanced/autowrap_ufuncify.py', 'examples/intermediate/sample.py', 'examples/intermediate/mplot2d.py', 'examples/intermediate/mplot3d.py', 'doc/src/modules/numeric-computation.rst'])
    elif import_module('matplotlib') is None:
        blacklist.extend(['examples/intermediate/mplot2d.py', 'examples/intermediate/mplot3d.py'])
    else:
        import matplotlib
        matplotlib.use('Agg')
    if ON_CI or import_module('pyglet') is None:
        blacklist.extend(['sympy/plotting/pygletplot'])
    if import_module('aesara') is None:
        blacklist.extend(['sympy/printing/aesaracode.py', 'doc/src/modules/numeric-computation.rst'])
    if import_module('cupy') is None:
        blacklist.extend(['doc/src/modules/numeric-computation.rst'])
    if import_module('jax') is None:
        blacklist.extend(['doc/src/modules/numeric-computation.rst'])
    if import_module('antlr4') is None:
        blacklist.extend(['sympy/parsing/autolev/__init__.py', 'sympy/parsing/latex/_parse_latex_antlr.py'])
    if import_module('lfortran') is None:
        blacklist.extend(['sympy/parsing/sym_expr.py'])
    if import_module('scipy') is None:
        blacklist.extend(['doc/src/guides/solving/solve-numerically.md', 'doc/src/guides/solving/solve-ode.md'])
    if import_module('numpy') is None:
        blacklist.extend(['doc/src/guides/solving/solve-ode.md', 'doc/src/guides/solving/solve-numerically.md'])
    blacklist.extend(['sympy/utilities/autowrap.py', 'examples/advanced/autowrap_integrators.py', 'examples/advanced/autowrap_ufuncify.py'])
    blacklist.extend(['sympy/conftest.py'])
    blacklist.extend(['sympy/utilities/tmpfiles.py', 'sympy/utilities/pytest.py', 'sympy/utilities/runtests.py', 'sympy/utilities/quality_unicode.py', 'sympy/utilities/randtest.py'])
    blacklist = convert_to_native_paths(blacklist)
    return blacklist