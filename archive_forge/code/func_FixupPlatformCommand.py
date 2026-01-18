import ast
import gyp.common
import gyp.simple_copy
import multiprocessing
import os.path
import re
import shlex
import signal
import subprocess
import sys
import threading
import traceback
from distutils.version import StrictVersion
from gyp.common import GypError
from gyp.common import OrderedSet
def FixupPlatformCommand(cmd):
    if sys.platform == 'win32':
        if type(cmd) is list:
            cmd = [re.sub('^cat ', 'type ', cmd[0])] + cmd[1:]
        else:
            cmd = re.sub('^cat ', 'type ', cmd)
    return cmd