import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _AddObjectiveCARCFlags(self, flags):
    if self._Test('CLANG_ENABLE_OBJC_ARC', 'YES', default='NO'):
        flags.append('-fobjc-arc')