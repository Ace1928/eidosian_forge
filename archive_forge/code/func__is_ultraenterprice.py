import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_ultraenterprice(self):
    return re.match('.*Ultra-Enterprise', self.info['uname_i']) is not None