import fnmatch
import logging
import os
import re
import sys
from . import DistlibException
from .compat import fsdecode
from .util import convert_path
def _exclude_pattern(self, pattern, anchor=True, prefix=None, is_regex=False):
    """Remove strings (presumably filenames) from 'files' that match
        'pattern'.

        Other parameters are the same as for 'include_pattern()', above.
        The list 'self.files' is modified in place. Return True if files are
        found.

        This API is public to allow e.g. exclusion of SCM subdirs, e.g. when
        packaging source distributions
        """
    found = False
    pattern_re = self._translate_pattern(pattern, anchor, prefix, is_regex)
    for f in list(self.files):
        if pattern_re.search(f):
            self.files.remove(f)
            found = True
    return found