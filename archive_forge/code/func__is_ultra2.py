import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_ultra2(self):
    return re.match('.*Ultra-2', self.info['uname_i']) is not None