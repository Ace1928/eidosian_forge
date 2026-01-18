from __future__ import absolute_import, print_function
import io
import os
import re
import sys
import time
import copy
import distutils.log
import textwrap
import hashlib
from distutils.core import Distribution, Extension
from distutils.command.build_ext import build_ext
from IPython.core import display
from IPython.core import magic_arguments
from IPython.core.magic import Magics, magics_class, cell_magic
from IPython.utils.text import dedent
from ..Shadow import __version__ as cython_version
from ..Compiler.Errors import CompileError
from .Inline import cython_inline, load_dynamic
from .Dependencies import cythonize
from ..Utils import captured_fd, print_captured
def _get_build_extension(self, extension=None, lib_dir=None, temp_dir=None, pgo_step_name=None, _build_ext=build_ext):
    self._clear_distutils_mkpath_cache()
    dist = Distribution()
    config_files = dist.find_config_files()
    try:
        config_files.remove('setup.cfg')
    except ValueError:
        pass
    dist.parse_config_files(config_files)
    if not temp_dir:
        temp_dir = lib_dir
    add_pgo_flags = self._add_pgo_flags
    if pgo_step_name:
        base_build_ext = _build_ext

        class _build_ext(_build_ext):

            def build_extensions(self):
                add_pgo_flags(self, pgo_step_name, temp_dir)
                base_build_ext.build_extensions(self)
    build_extension = _build_ext(dist)
    build_extension.finalize_options()
    if temp_dir:
        temp_dir = encode_fs(temp_dir)
        build_extension.build_temp = temp_dir
    if lib_dir:
        lib_dir = encode_fs(lib_dir)
        build_extension.build_lib = lib_dir
    if extension is not None:
        build_extension.extensions = [extension]
    return build_extension