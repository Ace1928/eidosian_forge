from __future__ import absolute_import
import hashlib
import inspect
import os
import re
import sys
from distutils.core import Distribution, Extension
from distutils.command.build_ext import build_ext
import Cython
from ..Compiler.Main import Context
from ..Compiler.Options import (default_options, CompilationOptions,
from ..Compiler.Visitor import CythonTransform, EnvTransform
from ..Compiler.ParseTreeTransforms import SkipDeclarations
from ..Compiler.TreeFragment import parse_from_strings
from ..Compiler.StringEncoding import _unicode
from .Dependencies import strip_string_literals, cythonize, cached_function
from ..Compiler import Pipeline
from ..Utils import get_cython_cache_dir
import cython as cython_module
class RuntimeCompiledFunction(object):

    def __init__(self, f):
        self._f = f
        self._body = get_body(inspect.getsource(f))

    def __call__(self, *args, **kwds):
        all = inspect.getcallargs(self._f, *args, **kwds)
        if IS_PY3:
            return cython_inline(self._body, locals=self._f.__globals__, globals=self._f.__globals__, **all)
        else:
            return cython_inline(self._body, locals=self._f.func_globals, globals=self._f.func_globals, **all)