import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_SUNW(self):
    return re.match('SUNW', self.info['uname_i']) is not None