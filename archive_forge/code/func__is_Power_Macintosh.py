import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_Power_Macintosh(self):
    return self.info['sysctl_hw']['hw.machine'] == 'Power Macintosh'