from __future__ import nested_scopes
import fnmatch
import os.path
from _pydev_runfiles.pydev_runfiles_coverage import start_coverage_support
from _pydevd_bundle.pydevd_constants import *  # @UnusedWildImport
import re
import time
def __match(self, filter_list, name):
    """ returns whether a test name matches the test filter """
    if filter_list is None:
        return 1
    for f in filter_list:
        if re.match(f, name):
            return 1
    return 0