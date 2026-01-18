import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetBundlePlugInsFolderPath(self):
    """Returns the qualified path to the bundle's plugins folder. E.g,
    Chromium.app/Contents/PlugIns. Only valid for bundles."""
    assert self._IsBundle()
    return os.path.join(self.GetBundleContentsFolderPath(), 'PlugIns')