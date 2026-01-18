import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _has_f00f_bug(self):
    return self.info[0]['f00f_bug'] == 'yes'