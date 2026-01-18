import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _has_sse3(self):
    return re.match('.*?\\bpni\\b', self.info[0]['flags']) is not None