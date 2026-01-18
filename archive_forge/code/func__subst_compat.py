import importlib.util
import os
import re
import string
import subprocess
import sys
import sysconfig
import functools
from .errors import DistutilsPlatformError, DistutilsByteCompileError
from ._modified import newer
from .spawn import spawn
from ._log import log
from distutils.util import byte_compile
def _subst_compat(s):
    """
    Replace shell/Perl-style variable substitution with
    format-style. For compatibility.
    """

    def _subst(match):
        return f'{{{match.group(1)}}}'
    repl = re.sub('\\$([a-zA-Z_][a-zA-Z_0-9]*)', _subst, s)
    if repl != s:
        import warnings
        warnings.warn('shell/Perl-style substitutions are deprecated', DeprecationWarning)
    return repl