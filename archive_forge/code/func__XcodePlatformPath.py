import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _XcodePlatformPath(self, configname=None):
    sdk_root = self._SdkRoot(configname)
    if sdk_root not in XcodeSettings._platform_path_cache:
        platform_path = self._GetSdkVersionInfoItem(sdk_root, '--show-sdk-platform-path')
        XcodeSettings._platform_path_cache[sdk_root] = platform_path
    return XcodeSettings._platform_path_cache[sdk_root]