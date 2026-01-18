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
def _add_dummy_mingwex_sym(self, c_sources):
    build_src = self.get_finalized_command('build_src').build_src
    build_clib = self.get_finalized_command('build_clib').build_clib
    objects = self.compiler.compile([os.path.join(build_src, 'gfortran_vs2003_hack.c')], output_dir=self.build_temp)
    self.compiler.create_static_lib(objects, '_gfortran_workaround', output_dir=build_clib, debug=self.debug)