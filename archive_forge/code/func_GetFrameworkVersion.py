import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetFrameworkVersion(self):
    """Returns the framework version of the current target. Only valid for
    bundles."""
    assert self._IsBundle()
    return self.GetPerTargetSetting('FRAMEWORK_VERSION', default='A')