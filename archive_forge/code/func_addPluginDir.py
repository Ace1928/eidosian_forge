from __future__ import annotations
import errno
import os
import sys
import warnings
from typing import AnyStr
from collections import OrderedDict
from typing import (
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
def addPluginDir():
    warnings.warn('twisted.python.util.addPluginDir is deprecated since Twisted 12.2.', DeprecationWarning, stacklevel=2)
    sys.path.extend(getPluginDirs())