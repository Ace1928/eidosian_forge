import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _has_sse(self):
    if self.is_Intel():
        return self.info[0]['Family'] == 6 and self.info[0]['Model'] in [7, 8, 9, 10, 11] or self.info[0]['Family'] == 15
    elif self.is_AMD():
        return self.info[0]['Family'] == 6 and self.info[0]['Model'] in [6, 7, 8, 10] or self.info[0]['Family'] == 15
    else:
        return False