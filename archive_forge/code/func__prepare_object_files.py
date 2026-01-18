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
def _prepare_object_files(self, build_ext):
    cc = self._cc
    dir_util.mkpath(os.path.join(build_ext.build_temp, *self.name.split('.')[:-1]))
    objects, _ = cc._compile_object_files(build_ext.build_temp)
    self.extra_objects = objects