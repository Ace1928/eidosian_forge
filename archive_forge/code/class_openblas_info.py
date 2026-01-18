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
class openblas_info(blas_info):
    section = 'openblas'
    dir_env_var = 'OPENBLAS'
    _lib_names = ['openblas']
    _require_symbols = []
    notfounderror = BlasNotFoundError

    @property
    def symbol_prefix(self):
        try:
            return self.cp.get(self.section, 'symbol_prefix')
        except NoOptionError:
            return ''

    @property
    def symbol_suffix(self):
        try:
            return self.cp.get(self.section, 'symbol_suffix')
        except NoOptionError:
            return ''

    def _calc_info(self):
        c = customized_ccompiler()
        lib_dirs = self.get_lib_dirs()
        opt = self.get_option_single('openblas_libs', 'libraries')
        openblas_libs = self.get_libs(opt, self._lib_names)
        info = self.check_libs(lib_dirs, openblas_libs, [])
        if c.compiler_type == 'msvc' and info is None:
            from numpy.distutils.fcompiler import new_fcompiler
            f = new_fcompiler(c_compiler=c)
            if f and f.compiler_type == 'gnu95':
                info = self.check_msvc_gfortran_libs(lib_dirs, openblas_libs)
                skip_symbol_check = True
        elif info:
            skip_symbol_check = False
            info['language'] = 'c'
        if info is None:
            return None
        extra_info = self.calc_extra_info()
        dict_append(info, **extra_info)
        if not (skip_symbol_check or self.check_symbols(info)):
            return None
        info['define_macros'] = [('HAVE_CBLAS', None)]
        if self.symbol_prefix:
            info['define_macros'] += [('BLAS_SYMBOL_PREFIX', self.symbol_prefix)]
        if self.symbol_suffix:
            info['define_macros'] += [('BLAS_SYMBOL_SUFFIX', self.symbol_suffix), ('OPENBLAS_ILP64_NAMING_SCHEME', None)]
        return info

    def calc_info(self):
        info = self._calc_info()
        if info is not None:
            self.set_info(**info)

    def check_msvc_gfortran_libs(self, library_dirs, libraries):
        library_paths = []
        for library in libraries:
            for library_dir in library_dirs:
                fullpath = os.path.join(library_dir, library + '.a')
                if os.path.isfile(fullpath):
                    library_paths.append(fullpath)
                    break
            else:
                return None
        basename = self.__class__.__name__
        tmpdir = os.path.join(os.getcwd(), 'build', basename)
        if not os.path.isdir(tmpdir):
            os.makedirs(tmpdir)
        info = {'library_dirs': [tmpdir], 'libraries': [basename], 'language': 'f77'}
        fake_lib_file = os.path.join(tmpdir, basename + '.fobjects')
        fake_clib_file = os.path.join(tmpdir, basename + '.cobjects')
        with open(fake_lib_file, 'w') as f:
            f.write('\n'.join(library_paths))
        with open(fake_clib_file, 'w') as f:
            pass
        return info

    def check_symbols(self, info):
        res = False
        c = customized_ccompiler()
        tmpdir = tempfile.mkdtemp()
        prototypes = '\n'.join(('void %s%s%s();' % (self.symbol_prefix, symbol_name, self.symbol_suffix) for symbol_name in self._require_symbols))
        calls = '\n'.join(('%s%s%s();' % (self.symbol_prefix, symbol_name, self.symbol_suffix) for symbol_name in self._require_symbols))
        s = textwrap.dedent('            %(prototypes)s\n            int main(int argc, const char *argv[])\n            {\n                %(calls)s\n                return 0;\n            }') % dict(prototypes=prototypes, calls=calls)
        src = os.path.join(tmpdir, 'source.c')
        out = os.path.join(tmpdir, 'a.out')
        try:
            extra_args = info['extra_link_args']
        except Exception:
            extra_args = []
        try:
            with open(src, 'w') as f:
                f.write(s)
            obj = c.compile([src], output_dir=tmpdir)
            try:
                c.link_executable(obj, out, libraries=info['libraries'], library_dirs=info['library_dirs'], extra_postargs=extra_args)
                res = True
            except distutils.ccompiler.LinkError:
                res = False
        finally:
            shutil.rmtree(tmpdir)
        return res