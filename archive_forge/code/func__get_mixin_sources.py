from setuptools import distutils as dutils
from setuptools.command import build_ext
from setuptools.extension import Extension
import os
import shutil
import sys
import tempfile
from numba.core import typing, sigutils
from numba.core.compiler_lock import global_compiler_lock
from numba.pycc.compiler import ModuleCompiler, ExportEntry
from numba.pycc.platform import Toolchain
from numba import cext
def _get_mixin_sources(self):
    here = os.path.dirname(__file__)
    mixin_sources = self._mixin_sources[:]
    if self._use_nrt:
        mixin_sources.append('../core/runtime/nrt.cpp')
    return [os.path.join(here, f) for f in mixin_sources]