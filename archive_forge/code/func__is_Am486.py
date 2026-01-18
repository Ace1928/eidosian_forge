import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_Am486(self):
    return self.is_AMD() and self.info[0]['Family'] == 4