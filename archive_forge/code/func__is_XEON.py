import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_XEON(self):
    return re.match('.*?XEON\\b', self.info[0]['model name'], re.IGNORECASE) is not None