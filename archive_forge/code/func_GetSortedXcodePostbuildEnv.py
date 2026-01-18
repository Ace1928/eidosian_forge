import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def GetSortedXcodePostbuildEnv(self):
    strip_save_file = self.xcode_settings.GetPerTargetSetting('CHROMIUM_STRIP_SAVE_FILE', '')
    return self.GetSortedXcodeEnv(additional_settings={'CHROMIUM_STRIP_SAVE_FILE': strip_save_file})