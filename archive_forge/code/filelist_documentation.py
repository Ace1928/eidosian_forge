import os, re
import fnmatch
import functools
from distutils.util import convert_path
from distutils.errors import DistutilsTemplateError, DistutilsInternalError
from distutils import log
Remove strings (presumably filenames) from 'files' that match
        'pattern'.  Other parameters are the same as for
        'include_pattern()', above.
        The list 'self.files' is modified in place.
        Return True if files are found, False otherwise.
        