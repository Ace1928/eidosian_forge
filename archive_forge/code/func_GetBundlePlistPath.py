import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetBundlePlistPath(self):
    """Returns the qualified path to the bundle's plist file. E.g.
    Chromium.app/Contents/Info.plist. Only valid for bundles."""
    assert self._IsBundle()
    if self.spec['type'] in ('executable', 'loadable_module') or self.IsIosFramework():
        return os.path.join(self.GetBundleContentsFolderPath(), 'Info.plist')
    else:
        return os.path.join(self.GetBundleContentsFolderPath(), 'Resources', 'Info.plist')