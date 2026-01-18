import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def GetArch(self, config):
    """Get architecture based on msvs_configuration_platform and
        msvs_target_platform. Returns either 'x86' or 'x64'."""
    configuration_platform = self.msvs_configuration_platform.get(config, '')
    platform = self.msvs_target_platform.get(config, '')
    if not platform:
        platform = configuration_platform
    return {'Win32': 'x86', 'x64': 'x64', 'ARM64': 'arm64'}.get(platform, 'x86')