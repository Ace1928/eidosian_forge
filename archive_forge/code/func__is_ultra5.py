import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_ultra5(self):
    return re.match('.*Ultra-5', self.info['uname_i']) is not None