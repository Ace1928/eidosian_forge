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
def _fix_lib_args(self, libraries, library_dirs, runtime_library_dirs):
    """Typecheck and fix up some of the arguments supplied to the
        'link_*' methods.  Specifically: ensure that all arguments are
        lists, and augment them with their permanent versions
        (eg. 'self.libraries' augments 'libraries').  Return a tuple with
        fixed versions of all arguments.
        """
    if libraries is None:
        libraries = self.libraries
    elif isinstance(libraries, (list, tuple)):
        libraries = list(libraries) + (self.libraries or [])
    else:
        raise TypeError("'libraries' (if supplied) must be a list of strings")
    if library_dirs is None:
        library_dirs = self.library_dirs
    elif isinstance(library_dirs, (list, tuple)):
        library_dirs = list(library_dirs) + (self.library_dirs or [])
    else:
        raise TypeError("'library_dirs' (if supplied) must be a list of strings")
    library_dirs += self.__class__.library_dirs
    if runtime_library_dirs is None:
        runtime_library_dirs = self.runtime_library_dirs
    elif isinstance(runtime_library_dirs, (list, tuple)):
        runtime_library_dirs = list(runtime_library_dirs) + (self.runtime_library_dirs or [])
    else:
        raise TypeError("'runtime_library_dirs' (if supplied) must be a list of strings")
    return (libraries, library_dirs, runtime_library_dirs)