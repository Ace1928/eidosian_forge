import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _IsXCTest(self):
    return int(self.spec.get('mac_xctest_bundle', 0)) != 0