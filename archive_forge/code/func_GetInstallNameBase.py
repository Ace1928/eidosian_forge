import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetInstallNameBase(self):
    """Return DYLIB_INSTALL_NAME_BASE for this target."""
    if self.spec['type'] != 'shared_library' and (self.spec['type'] != 'loadable_module' or self._IsBundle()):
        return None
    install_base = self.GetPerTargetSetting('DYLIB_INSTALL_NAME_BASE', default='/Library/Frameworks' if self._IsBundle() else '/usr/local/lib')
    return install_base