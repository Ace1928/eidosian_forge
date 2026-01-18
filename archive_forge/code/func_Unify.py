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
def Unify(items):
    """Removes duplicate elements from items, keeping the first element."""
    seen = {}
    return [seen.setdefault(e, e) for e in items if e not in seen]