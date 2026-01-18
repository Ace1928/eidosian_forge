import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_ultra5_10(self):
    return re.match('.*Ultra-5_10', self.info['uname_i']) is not None