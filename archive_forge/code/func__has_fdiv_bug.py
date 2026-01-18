import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _has_fdiv_bug(self):
    return self.info[0]['fdiv_bug'] == 'yes'