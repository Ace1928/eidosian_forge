import os
import sys
import re
import shlex
import itertools
from . import sysconfig
from ._modified import newer
from .ccompiler import CCompiler, gen_preprocess_options, gen_lib_options
from .errors import DistutilsExecError, CompileError, LibError, LinkError
from ._log import log
from ._macos_compat import compiler_fixup
def _split_env(cmd):
    """
    For macOS, split command into 'env' portion (if any)
    and the rest of the linker command.

    >>> _split_env(['a', 'b', 'c'])
    ([], ['a', 'b', 'c'])
    >>> _split_env(['/usr/bin/env', 'A=3', 'gcc'])
    (['/usr/bin/env', 'A=3'], ['gcc'])
    """
    pivot = 0
    if os.path.basename(cmd[0]) == 'env':
        pivot = 1
        while '=' in cmd[pivot]:
            pivot += 1
    return (cmd[:pivot], cmd[pivot:])