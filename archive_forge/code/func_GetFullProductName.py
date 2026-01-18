import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetFullProductName(self):
    """Returns FULL_PRODUCT_NAME."""
    if self._IsBundle():
        return self.GetWrapperName()
    else:
        return self._GetStandaloneBinaryPath()