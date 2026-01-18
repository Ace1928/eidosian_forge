import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def IsBinaryOutputFormat(self, configname):
    default = 'binary' if self.isIOS else 'xml'
    format = self.xcode_settings[configname].get('INFOPLIST_OUTPUT_FORMAT', default)
    return format == 'binary'