import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_sparcstation5(self):
    return re.match('.*SPARCstation-5', self.info['uname_i']) is not None