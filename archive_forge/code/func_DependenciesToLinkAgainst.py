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
def DependenciesToLinkAgainst(self, targets):
    """
    Returns a list of dependency targets that are linked into this target.
    """
    return self._LinkDependenciesInternal(targets, True)