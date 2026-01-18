import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_Itanium(self):
    return re.match('.*?Itanium\\b', self.info[0]['family']) is not None