import os
import sys
import re
from pathlib import Path
from distutils.sysconfig import get_python_lib
from distutils.fancy_getopt import FancyGetopt
from distutils.errors import DistutilsModuleError, \
from distutils.util import split_quoted, strtobool
from numpy.distutils.ccompiler import CCompiler, gen_lib_options
from numpy.distutils import log
from numpy.distutils.misc_util import is_string, all_strings, is_sequence, \
from numpy.distutils.exec_command import find_executable
from numpy.distutils import _shell_utils
from .environment import EnvironmentConfig
def _find_existing_fcompiler(compiler_types, osname=None, platform=None, requiref90=False, c_compiler=None):
    from numpy.distutils.core import get_distribution
    dist = get_distribution(always=True)
    for compiler_type in compiler_types:
        v = None
        try:
            c = new_fcompiler(plat=platform, compiler=compiler_type, c_compiler=c_compiler)
            c.customize(dist)
            v = c.get_version()
            if requiref90 and c.compiler_f90 is None:
                v = None
                new_compiler = c.suggested_f90_compiler
                if new_compiler:
                    log.warn('Trying %r compiler as suggested by %r compiler for f90 support.' % (compiler_type, new_compiler))
                    c = new_fcompiler(plat=platform, compiler=new_compiler, c_compiler=c_compiler)
                    c.customize(dist)
                    v = c.get_version()
                    if v is not None:
                        compiler_type = new_compiler
            if requiref90 and c.compiler_f90 is None:
                raise ValueError('%s does not support compiling f90 codes, skipping.' % c.__class__.__name__)
        except DistutilsModuleError:
            log.debug("_find_existing_fcompiler: compiler_type='%s' raised DistutilsModuleError", compiler_type)
        except CompilerNotFound:
            log.debug("_find_existing_fcompiler: compiler_type='%s' not found", compiler_type)
        if v is not None:
            return compiler_type
    return None