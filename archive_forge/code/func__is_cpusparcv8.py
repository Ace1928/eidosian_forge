import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_cpusparcv8(self):
    return self.info['processor'] == 'sparcv8'