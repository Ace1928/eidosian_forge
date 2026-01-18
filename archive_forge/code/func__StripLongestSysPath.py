from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import sys
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_attr_os
import six
from six.moves import range
from six.moves import urllib
def _StripLongestSysPath(path):
    python_paths = sorted(_SysPath(), key=len, reverse=True)
    for python_path in python_paths:
        prefix = python_path + os.path.sep
        if path.startswith(prefix):
            return path[len(prefix):]
    return path