import os
import sys
import subprocess
import re
import textwrap
import numpy.distutils.ccompiler  # noqa: F401
from numpy.distutils import log
import distutils.cygwinccompiler
from distutils.unixccompiler import UnixCCompiler
from distutils.msvccompiler import get_build_version as get_build_msvc_version
from distutils.errors import UnknownFileError
from numpy.distutils.misc_util import (msvc_runtime_library,
class Mingw32CCompiler(distutils.cygwinccompiler.CygwinCCompiler):
    """ A modified MingW32 compiler compatible with an MSVC built Python.

    """
    compiler_type = 'mingw32'

    def __init__(self, verbose=0, dry_run=0, force=0):
        distutils.cygwinccompiler.CygwinCCompiler.__init__(self, verbose, dry_run, force)
        build_import_library()
        msvcr_success = build_msvcr_library()
        msvcr_dbg_success = build_msvcr_library(debug=True)
        if msvcr_success or msvcr_dbg_success:
            self.define_macro('NPY_MINGW_USE_CUSTOM_MSVCR')
        msvcr_version = msvc_runtime_version()
        if msvcr_version:
            self.define_macro('__MSVCRT_VERSION__', '0x%04i' % msvcr_version)
        if get_build_architecture() == 'AMD64':
            self.set_executables(compiler='gcc -g -DDEBUG -DMS_WIN64 -O0 -Wall', compiler_so='gcc -g -DDEBUG -DMS_WIN64 -O0 -Wall -Wstrict-prototypes', linker_exe='gcc -g', linker_so='gcc -g -shared')
        else:
            self.set_executables(compiler='gcc -O2 -Wall', compiler_so='gcc -O2 -Wall -Wstrict-prototypes', linker_exe='g++ ', linker_so='g++ -shared')
        self.compiler_cxx = ['g++']
        return

    def link(self, target_desc, objects, output_filename, output_dir, libraries, library_dirs, runtime_library_dirs, export_symbols=None, debug=0, extra_preargs=None, extra_postargs=None, build_temp=None, target_lang=None):
        runtime_library = msvc_runtime_library()
        if runtime_library:
            if not libraries:
                libraries = []
            libraries.append(runtime_library)
        args = (self, target_desc, objects, output_filename, output_dir, libraries, library_dirs, runtime_library_dirs, None, debug, extra_preargs, extra_postargs, build_temp, target_lang)
        func = UnixCCompiler.link
        func(*args[:func.__code__.co_argcount])
        return

    def object_filenames(self, source_filenames, strip_dir=0, output_dir=''):
        if output_dir is None:
            output_dir = ''
        obj_names = []
        for src_name in source_filenames:
            base, ext = os.path.splitext(os.path.normcase(src_name))
            drv, base = os.path.splitdrive(base)
            if drv:
                base = base[1:]
            if ext not in self.src_extensions + ['.rc', '.res']:
                raise UnknownFileError("unknown file type '%s' (from '%s')" % (ext, src_name))
            if strip_dir:
                base = os.path.basename(base)
            if ext == '.res' or ext == '.rc':
                obj_names.append(os.path.join(output_dir, base + ext + self.obj_extension))
            else:
                obj_names.append(os.path.join(output_dir, base + self.obj_extension))
        return obj_names