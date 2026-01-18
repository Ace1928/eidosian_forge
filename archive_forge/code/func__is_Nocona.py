import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_Nocona(self):
    return self.is_Intel() and (self.info[0]['cpu family'] == '6' or self.info[0]['cpu family'] == '15') and (self.has_sse3() and (not self.has_ssse3())) and (re.match('.*?\\blm\\b', self.info[0]['flags']) is not None)