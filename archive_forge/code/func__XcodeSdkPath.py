import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _XcodeSdkPath(self, sdk_root):
    if sdk_root not in XcodeSettings._sdk_path_cache:
        sdk_path = self._GetSdkVersionInfoItem(sdk_root, '--show-sdk-path')
        XcodeSettings._sdk_path_cache[sdk_root] = sdk_path
        if sdk_root:
            XcodeSettings._sdk_root_cache[sdk_path] = sdk_root
    return XcodeSettings._sdk_path_cache[sdk_root]