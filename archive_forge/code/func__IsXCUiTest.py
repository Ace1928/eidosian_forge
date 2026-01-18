import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _IsXCUiTest(self):
    return int(self.spec.get('mac_xcuitest_bundle', 0)) != 0