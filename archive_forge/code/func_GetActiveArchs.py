import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetActiveArchs(self, configname):
    """Returns the architectures this target should be built for."""
    config_settings = self.xcode_settings[configname]
    xcode_archs_default = GetXcodeArchsDefault()
    return xcode_archs_default.ActiveArchs(config_settings.get('ARCHS'), config_settings.get('VALID_ARCHS'), config_settings.get('SDKROOT'))