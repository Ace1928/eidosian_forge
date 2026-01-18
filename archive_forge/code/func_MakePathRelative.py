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
def MakePathRelative(to_file, fro_file, item):
    if to_file == fro_file or exception_re.match(item):
        return item
    else:
        ret = os.path.normpath(os.path.join(gyp.common.RelativePath(os.path.dirname(fro_file), os.path.dirname(to_file)), item)).replace('\\', '/')
        if item.endswith('/'):
            ret += '/'
        return ret