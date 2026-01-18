import os
import subprocess
from glob import glob
from distutils.dep_util import newer_group
from distutils.command.build_ext import build_ext as old_build_ext
from distutils.errors import DistutilsFileError, DistutilsSetupError,\
from distutils.file_util import copy_file
from numpy.distutils import log
from numpy.distutils.exec_command import filepath_from_subprocess_output
from numpy.distutils.system_info import combine_paths
from numpy.distutils.misc_util import (
from numpy.distutils.command.config_compiler import show_fortran_compilers
from numpy.distutils.ccompiler_opt import new_ccompiler_opt, CCompilerOpt
def _process_unlinkable_fobjects(self, objects, libraries, fcompiler, library_dirs, unlinkable_fobjects):
    libraries = list(libraries)
    objects = list(objects)
    unlinkable_fobjects = list(unlinkable_fobjects)
    for lib in libraries[:]:
        for libdir in library_dirs:
            fake_lib = os.path.join(libdir, lib + '.fobjects')
            if os.path.isfile(fake_lib):
                libraries.remove(lib)
                with open(fake_lib) as f:
                    unlinkable_fobjects.extend(f.read().splitlines())
                c_lib = os.path.join(libdir, lib + '.cobjects')
                with open(c_lib) as f:
                    objects.extend(f.read().splitlines())
    if unlinkable_fobjects:
        fobjects = [os.path.abspath(obj) for obj in unlinkable_fobjects]
        wrapped = fcompiler.wrap_unlinkable_objects(fobjects, output_dir=self.build_temp, extra_dll_dir=self.extra_dll_dir)
        objects.extend(wrapped)
    return (objects, libraries)