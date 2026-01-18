import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _VariableMapping(self, sdkroot):
    """Returns the dictionary of variable mapping depending on the SDKROOT."""
    sdkroot = sdkroot.lower()
    if 'iphoneos' in sdkroot:
        return self._archs['ios']
    elif 'iphonesimulator' in sdkroot:
        return self._archs['iossim']
    else:
        return self._archs['mac']