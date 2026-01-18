import re
import os
import sys
import warnings
import platform
import tempfile
import hashlib
import base64
import subprocess
from subprocess import Popen, PIPE, STDOUT
from numpy.distutils.exec_command import filepath_from_subprocess_output
from numpy.distutils.fcompiler import FCompiler
from distutils.version import LooseVersion
def _link_wrapper_lib(self, objects, output_dir, extra_dll_dir, chained_dlls, is_archive):
    """Create a wrapper shared library for the given objects

        Return an MSVC-compatible lib
        """
    c_compiler = self.c_compiler
    if c_compiler.compiler_type != 'msvc':
        raise ValueError('This method only supports MSVC')
    object_hash = self._hash_files(list(objects) + list(chained_dlls))
    if is_win64():
        tag = 'win_amd64'
    else:
        tag = 'win32'
    basename = 'lib' + os.path.splitext(os.path.basename(objects[0]))[0][:8]
    root_name = basename + '.' + object_hash + '.gfortran-' + tag
    dll_name = root_name + '.dll'
    def_name = root_name + '.def'
    lib_name = root_name + '.lib'
    dll_path = os.path.join(extra_dll_dir, dll_name)
    def_path = os.path.join(output_dir, def_name)
    lib_path = os.path.join(output_dir, lib_name)
    if os.path.isfile(lib_path):
        return (lib_path, dll_path)
    if is_archive:
        objects = ['-Wl,--whole-archive'] + list(objects) + ['-Wl,--no-whole-archive']
    self.link_shared_object(objects, dll_name, output_dir=extra_dll_dir, extra_postargs=list(chained_dlls) + ['-Wl,--allow-multiple-definition', '-Wl,--output-def,' + def_path, '-Wl,--export-all-symbols', '-Wl,--enable-auto-import', '-static', '-mlong-double-64'])
    if is_win64():
        specifier = '/MACHINE:X64'
    else:
        specifier = '/MACHINE:X86'
    lib_args = ['/def:' + def_path, '/OUT:' + lib_path, specifier]
    if not c_compiler.initialized:
        c_compiler.initialize()
    c_compiler.spawn([c_compiler.lib] + lib_args)
    return (lib_path, dll_path)