import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _has_mmx(self):
    if self.is_Intel():
        return self.info[0]['Family'] == 5 and self.info[0]['Model'] == 4 or self.info[0]['Family'] in [6, 15]
    elif self.is_AMD():
        return self.info[0]['Family'] in [5, 6, 15]
    else:
        return False