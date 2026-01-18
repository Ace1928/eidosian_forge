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
def IsPathSection(section):
    while section and section[-1:] in '=+?!':
        section = section[:-1]
    if section in path_sections:
        return True
    if '_' in section:
        tail = section[-6:]
        if tail[-1] == 's':
            tail = tail[:-1]
        if tail[-5:] in ('_file', '_path'):
            return True
        return tail[-4:] == '_dir'
    return False