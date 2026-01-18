import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _has_ssse3(self):
    return re.match('.*?\\bssse3\\b', self.info[0]['flags']) is not None