import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetBundleSharedSupportFolderPath(self):
    """Returns the qualified path to the bundle's shared support folder. E.g,
    Chromium.app/Contents/SharedSupport. Only valid for bundles."""
    assert self._IsBundle()
    if self.spec['type'] == 'shared_library':
        return self.GetBundleResourceFolder()
    else:
        return os.path.join(self.GetBundleContentsFolderPath(), 'SharedSupport')