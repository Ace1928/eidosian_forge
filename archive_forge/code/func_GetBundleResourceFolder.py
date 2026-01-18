import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetBundleResourceFolder(self):
    """Returns the qualified path to the bundle's resource folder. E.g.
    Chromium.app/Contents/Resources. Only valid for bundles."""
    assert self._IsBundle()
    if self.isIOS:
        return self.GetBundleContentsFolderPath()
    return os.path.join(self.GetBundleContentsFolderPath(), 'Resources')