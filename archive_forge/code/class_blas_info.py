import sys
import os
import re
import copy
import warnings
import subprocess
import textwrap
from glob import glob
from functools import reduce
from configparser import NoOptionError
from configparser import RawConfigParser as ConfigParser
from distutils.errors import DistutilsError
from distutils.dist import Distribution
import sysconfig
from numpy.distutils import log
from distutils.util import get_platform
from numpy.distutils.exec_command import (
from numpy.distutils.misc_util import (is_sequence, is_string,
from numpy.distutils.command.config import config as cmd_config
from numpy.distutils import customized_ccompiler as _customized_ccompiler
from numpy.distutils import _shell_utils
import distutils.ccompiler
import tempfile
import shutil
import platform
class blas_info(system_info):
    section = 'blas'
    dir_env_var = 'BLAS'
    _lib_names = ['blas']
    notfounderror = BlasNotFoundError

    def calc_info(self):
        lib_dirs = self.get_lib_dirs()
        opt = self.get_option_single('blas_libs', 'libraries')
        blas_libs = self.get_libs(opt, self._lib_names)
        info = self.check_libs(lib_dirs, blas_libs, [])
        if info is None:
            return
        else:
            info['include_dirs'] = self.get_include_dirs()
        if platform.system() == 'Windows':
            info['language'] = 'f77'
            cblas_info_obj = cblas_info()
            cblas_opt = cblas_info_obj.get_option_single('cblas_libs', 'libraries')
            cblas_libs = cblas_info_obj.get_libs(cblas_opt, None)
            if cblas_libs:
                info['libraries'] = cblas_libs + blas_libs
                info['define_macros'] = [('HAVE_CBLAS', None)]
        else:
            lib = self.get_cblas_libs(info)
            if lib is not None:
                info['language'] = 'c'
                info['libraries'] = lib
                info['define_macros'] = [('HAVE_CBLAS', None)]
        self.set_info(**info)

    def get_cblas_libs(self, info):
        """ Check whether we can link with CBLAS interface

        This method will search through several combinations of libraries
        to check whether CBLAS is present:

        1. Libraries in ``info['libraries']``, as is
        2. As 1. but also explicitly adding ``'cblas'`` as a library
        3. As 1. but also explicitly adding ``'blas'`` as a library
        4. Check only library ``'cblas'``
        5. Check only library ``'blas'``

        Parameters
        ----------
        info : dict
           system information dictionary for compilation and linking

        Returns
        -------
        libraries : list of str or None
            a list of libraries that enables the use of CBLAS interface.
            Returns None if not found or a compilation error occurs.

            Since 1.17 returns a list.
        """
        c = customized_ccompiler()
        tmpdir = tempfile.mkdtemp()
        s = textwrap.dedent('            #include <cblas.h>\n            int main(int argc, const char *argv[])\n            {\n                double a[4] = {1,2,3,4};\n                double b[4] = {5,6,7,8};\n                return cblas_ddot(4, a, 1, b, 1) > 10;\n            }')
        src = os.path.join(tmpdir, 'source.c')
        try:
            with open(src, 'w') as f:
                f.write(s)
            try:
                obj = c.compile([src], output_dir=tmpdir, include_dirs=self.get_include_dirs())
            except (distutils.ccompiler.CompileError, distutils.ccompiler.LinkError):
                return None
            for libs in [info['libraries'], ['cblas'] + info['libraries'], ['blas'] + info['libraries'], ['cblas'], ['blas']]:
                try:
                    c.link_executable(obj, os.path.join(tmpdir, 'a.out'), libraries=libs, library_dirs=info['library_dirs'], extra_postargs=info.get('extra_link_args', []))
                    return libs
                except distutils.ccompiler.LinkError:
                    pass
        finally:
            shutil.rmtree(tmpdir)
        return None