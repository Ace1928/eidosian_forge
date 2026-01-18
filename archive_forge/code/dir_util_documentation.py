import os
import errno
from distutils.errors import DistutilsFileError, DistutilsInternalError
from distutils import log
Take the full path 'path', and make it a relative path.

    This is useful to make 'path' the second argument to os.path.join().
    