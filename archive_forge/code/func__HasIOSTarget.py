import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _HasIOSTarget(targets):
    """Returns true if any target contains the iOS specific key
  IPHONEOS_DEPLOYMENT_TARGET."""
    for target_dict in targets.values():
        for config in target_dict['configurations'].values():
            if config.get('xcode_settings', {}).get('IPHONEOS_DEPLOYMENT_TARGET'):
                return True
    return False