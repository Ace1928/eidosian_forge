import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _has_3dnow(self):
    return self.is_AMD() and self.info[0]['Family'] in [5, 6, 15]