from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import pkgutil
import sys
import tempfile
import gslib.exception  # pylint: disable=g-import-not-at-top
from gslib.utils.version_check import check_python_version_support
def GetGsutilVersionModifiedTime():
    """Returns unix timestamp of when the VERSION file was last modified."""
    if not VERSION_FILE:
        return 0
    return int(os.path.getmtime(VERSION_FILE))