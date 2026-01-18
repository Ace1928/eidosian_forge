import sys
import os
import re
import warnings
from .errors import (
from .spawn import spawn
from .file_util import move_file
from .dir_util import mkpath
from ._modified import newer_group
from .util import split_quoted, execute
from ._log import log
def _fix_compile_args(self, output_dir, macros, include_dirs):
    """Typecheck and fix-up some of the arguments to the 'compile()'
        method, and return fixed-up values.  Specifically: if 'output_dir'
        is None, replaces it with 'self.output_dir'; ensures that 'macros'
        is a list, and augments it with 'self.macros'; ensures that
        'include_dirs' is a list, and augments it with 'self.include_dirs'.
        Guarantees that the returned values are of the correct type,
        i.e. for 'output_dir' either string or None, and for 'macros' and
        'include_dirs' either list or None.
        """
    if output_dir is None:
        output_dir = self.output_dir
    elif not isinstance(output_dir, str):
        raise TypeError("'output_dir' must be a string or None")
    if macros is None:
        macros = self.macros
    elif isinstance(macros, list):
        macros = macros + (self.macros or [])
    else:
        raise TypeError("'macros' (if supplied) must be a list of tuples")
    if include_dirs is None:
        include_dirs = list(self.include_dirs)
    elif isinstance(include_dirs, (list, tuple)):
        include_dirs = list(include_dirs) + (self.include_dirs or [])
    else:
        raise TypeError("'include_dirs' (if supplied) must be a list of strings")
    include_dirs += self.__class__.include_dirs
    return (output_dir, macros, include_dirs)