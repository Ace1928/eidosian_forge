import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def GetSortedXcodeEnv(self, additional_settings=None):
    return gyp.xcode_emulation.GetSortedXcodeEnv(self.xcode_settings, '$(abs_builddir)', os.path.join('$(abs_srcdir)', self.path), '$(BUILDTYPE)', additional_settings)