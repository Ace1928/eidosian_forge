import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _BuildMachineOSBuild(self):
    return GetStdout(['sw_vers', '-buildVersion'])