import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _XcodeIOSDeviceFamily(self, configname):
    family = self.xcode_settings[configname].get('TARGETED_DEVICE_FAMILY', '1')
    return [int(x) for x in family.split(',')]