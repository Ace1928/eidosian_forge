import os
import signal
import subprocess
import sys
import textwrap
import warnings
from distutils.command.config import config as old_config
from distutils.command.config import LANG_EXT
from distutils import log
from distutils.file_util import copy_file
from distutils.ccompiler import CompileError, LinkError
import distutils
from numpy.distutils.exec_command import filepath_from_subprocess_output
from numpy.distutils.mingw32ccompiler import generate_manifest
from numpy.distutils.command.autodist import (check_gcc_function_attribute,
def _wrap_method(self, mth, lang, args):
    from distutils.ccompiler import CompileError
    from distutils.errors import DistutilsExecError
    save_compiler = self.compiler
    if lang in ['f77', 'f90']:
        self.compiler = self.fcompiler
    if self.compiler is None:
        raise CompileError('%s compiler is not set' % (lang,))
    try:
        ret = mth(*(self,) + args)
    except (DistutilsExecError, CompileError) as e:
        self.compiler = save_compiler
        raise CompileError from e
    self.compiler = save_compiler
    return ret