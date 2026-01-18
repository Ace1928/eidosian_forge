import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_AthlonMP(self):
    return re.match('.*?Athlon\\(tm\\) MP\\b', self.info[0]['model name']) is not None