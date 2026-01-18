import os
import re
import sys
import platform
import shlex
import time
import subprocess
from copy import copy
from pathlib import Path
from distutils import ccompiler
from distutils.ccompiler import (
from distutils.errors import (
from distutils.sysconfig import customize_compiler
from distutils.version import LooseVersion
from numpy.distutils import log
from numpy.distutils.exec_command import (
from numpy.distutils.misc_util import cyg2win32, is_sequence, mingw32, \
import threading
def CCompiler_spawn(self, cmd, display=None, env=None):
    """
    Execute a command in a sub-process.

    Parameters
    ----------
    cmd : str
        The command to execute.
    display : str or sequence of str, optional
        The text to add to the log file kept by `numpy.distutils`.
        If not given, `display` is equal to `cmd`.
    env : a dictionary for environment variables, optional

    Returns
    -------
    None

    Raises
    ------
    DistutilsExecError
        If the command failed, i.e. the exit status was not 0.

    """
    env = env if env is not None else dict(os.environ)
    if display is None:
        display = cmd
        if is_sequence(display):
            display = ' '.join(list(display))
    log.info(display)
    try:
        if self.verbose:
            subprocess.check_output(cmd, env=env)
        else:
            subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=env)
    except subprocess.CalledProcessError as exc:
        o = exc.output
        s = exc.returncode
    except OSError as e:
        o = f'\n\n{e}\n\n\n'
        try:
            o = o.encode(sys.stdout.encoding)
        except AttributeError:
            o = o.encode('utf8')
        s = 127
    else:
        return None
    if is_sequence(cmd):
        cmd = ' '.join(list(cmd))
    if self.verbose:
        forward_bytes_to_stdout(o)
    if re.search(b'Too many open files', o):
        msg = '\nTry rerunning setup command until build succeeds.'
    else:
        msg = ''
    raise DistutilsExecError('Command "%s" failed with exit status %d%s' % (cmd, s, msg))