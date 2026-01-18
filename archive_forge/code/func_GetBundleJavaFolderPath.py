import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetBundleJavaFolderPath(self):
    """Returns the qualified path to the bundle's Java resource folder.
    E.g. Chromium.app/Contents/Resources/Java. Only valid for bundles."""
    assert self._IsBundle()
    return os.path.join(self.GetBundleResourceFolder(), 'Java')