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
def IsStrCanonicalInt(string):
    """Returns True if |string| is in its canonical integer form.

  The canonical form is such that str(int(string)) == string.
  """
    if type(string) is str:
        if string:
            if string == '0':
                return True
            if string[0] == '-':
                string = string[1:]
                if not string:
                    return False
            if '1' <= string[0] <= '9':
                return string.isdigit()
    return False