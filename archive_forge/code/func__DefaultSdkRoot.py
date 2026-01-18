import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _DefaultSdkRoot(self):
    """Returns the default SDKROOT to use.

    Prior to version 5.0.0, if SDKROOT was not explicitly set in the Xcode
    project, then the environment variable was empty. Starting with this
    version, Xcode uses the name of the newest SDK installed.
    """
    xcode_version, _ = XcodeVersion()
    if xcode_version < '0500':
        return ''
    default_sdk_path = self._XcodeSdkPath('')
    default_sdk_root = XcodeSettings._sdk_root_cache.get(default_sdk_path)
    if default_sdk_root:
        return default_sdk_root
    try:
        all_sdks = GetStdout(['xcodebuild', '-showsdks'])
    except GypError:
        return ''
    for line in all_sdks.splitlines():
        items = line.split()
        if len(items) >= 3 and items[-2] == '-sdk':
            sdk_root = items[-1]
            sdk_path = self._XcodeSdkPath(sdk_root)
            if sdk_path == default_sdk_path:
                return sdk_root
    return ''