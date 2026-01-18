import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def _InstallImmediately(self):
    return self.toolset == 'target' and self.flavor == 'mac' and (self.type in ('static_library', 'executable', 'shared_library', 'loadable_module'))