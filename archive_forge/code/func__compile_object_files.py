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
@global_compiler_lock
def _compile_object_files(self, build_dir):
    compiler = ModuleCompiler(self._export_entries, self._basename, self._use_nrt, cpu_name=self._target_cpu)
    compiler.external_init_function = self._init_function
    temp_obj = os.path.join(build_dir, os.path.splitext(self._output_file)[0] + '.o')
    log.info("generating LLVM code for '%s' into %s", self._basename, temp_obj)
    compiler.write_native_object(temp_obj, wrap=True)
    return ([temp_obj], compiler.dll_exports)