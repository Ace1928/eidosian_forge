import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_ultra80(self):
    return re.match('.*Ultra-80', self.info['uname_i']) is not None