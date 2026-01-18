import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _WarnUnimplemented(self, test_key):
    if test_key in self._Settings():
        print('Warning: Ignoring not yet implemented key "%s".' % test_key)