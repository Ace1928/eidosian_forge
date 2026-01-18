import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _Appendf(self, lst, test_key, format_str, default=None):
    if test_key in self._Settings():
        lst.append(format_str % str(self._Settings()[test_key]))
    elif default:
        lst.append(format_str % str(default))