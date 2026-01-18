import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_sunfire(self):
    return re.match('.*Sun-Fire', self.info['uname_i']) is not None