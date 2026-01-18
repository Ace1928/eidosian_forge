import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _GetBundleBinaryPath(self):
    """Returns the name of the bundle binary of by this target.
    E.g. Chromium.app/Contents/MacOS/Chromium. Only valid for bundles."""
    assert self._IsBundle()
    return os.path.join(self.GetBundleExecutableFolderPath(), self.GetExecutableName())