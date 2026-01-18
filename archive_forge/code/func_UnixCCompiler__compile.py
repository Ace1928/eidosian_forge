import os
import sys
import subprocess
import shlex
from distutils.errors import CompileError, DistutilsExecError, LibError
from distutils.unixccompiler import UnixCCompiler
from numpy.distutils.ccompiler import replace_method
from numpy.distutils.misc_util import _commandline_dep_string
from numpy.distutils import log
def UnixCCompiler__compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
    """Compile a single source files with a Unix-style compiler."""
    ccomp = self.compiler_so
    if ccomp[0] == 'aCC':
        if '-Ae' in ccomp:
            ccomp.remove('-Ae')
        if '-Aa' in ccomp:
            ccomp.remove('-Aa')
        ccomp += ['-AA']
        self.compiler_so = ccomp
    if 'OPT' in os.environ:
        from sysconfig import get_config_vars
        opt = shlex.join(shlex.split(os.environ['OPT']))
        gcv_opt = shlex.join(shlex.split(get_config_vars('OPT')[0]))
        ccomp_s = shlex.join(self.compiler_so)
        if opt not in ccomp_s:
            ccomp_s = ccomp_s.replace(gcv_opt, opt)
            self.compiler_so = shlex.split(ccomp_s)
        llink_s = shlex.join(self.linker_so)
        if opt not in llink_s:
            self.linker_so = self.linker_so + shlex.split(opt)
    display = '%s: %s' % (os.path.basename(self.compiler_so[0]), src)
    if getattr(self, '_auto_depends', False):
        deps = ['-MMD', '-MF', obj + '.d']
    else:
        deps = []
    try:
        self.spawn(self.compiler_so + cc_args + [src, '-o', obj] + deps + extra_postargs, display=display)
    except DistutilsExecError as e:
        msg = str(e)
        raise CompileError(msg) from None
    if deps:
        if sys.platform == 'zos':
            subprocess.check_output(['chtag', '-tc', 'IBM1047', obj + '.d'])
        with open(obj + '.d', 'a') as f:
            f.write(_commandline_dep_string(cc_args, extra_postargs, pp_opts))