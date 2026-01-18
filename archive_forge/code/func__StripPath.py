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
def _StripPath(path):
    """Removes common elements (sys.path, common SDK directories) from path."""
    return _StripCommonDir(os.path.normpath(_StripLongestSysPath(path)))