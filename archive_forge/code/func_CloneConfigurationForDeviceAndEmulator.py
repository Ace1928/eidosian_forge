import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def CloneConfigurationForDeviceAndEmulator(target_dicts):
    """If |target_dicts| contains any iOS targets, automatically create -iphoneos
  targets for iOS device builds."""
    if _HasIOSTarget(target_dicts):
        return _AddIOSDeviceConfigurations(target_dicts)
    return target_dicts