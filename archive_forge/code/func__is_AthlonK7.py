import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_AthlonK7(self):
    return re.match('.*?AMD-K7', self.info[0]['model name']) is not None