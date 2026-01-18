import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_sparcv9(self):
    return self.info['isainfo_n'] == 'sparcv9'