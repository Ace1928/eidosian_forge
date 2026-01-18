import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _GetSdkVersionInfoItem(self, sdk, infoitem):
    try:
        return GetStdoutQuiet(['xcrun', '--sdk', sdk, infoitem])
    except GypError:
        pass