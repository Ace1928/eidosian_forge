import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetBundleXPCServicesFolderPath(self):
    """Returns the qualified path to the bundle's XPC services folder. E.g,
    Chromium.app/Contents/XPCServices. Only valid for bundles."""
    assert self._IsBundle()
    return os.path.join(self.GetBundleContentsFolderPath(), 'XPCServices')