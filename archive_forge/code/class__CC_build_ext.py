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
class _CC_build_ext(_orig_build_ext):

    def build_extension(self, ext):
        if isinstance(ext, _CCExtension):
            ext._prepare_object_files(self)
        _orig_build_ext.build_extension(self, ext)