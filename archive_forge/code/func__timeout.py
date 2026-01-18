import os
import sys
import platform
import inspect
import traceback
import pdb
import re
import linecache
import time
from fnmatch import fnmatch
from timeit import default_timer as clock
import doctest as pdoctest  # avoid clashing with our doctest() function
from doctest import DocTestFinder, DocTestRunner
import random
import subprocess
import shutil
import signal
import stat
import tempfile
import warnings
from contextlib import contextmanager
from inspect import unwrap
from sympy.core.cache import clear_cache
from sympy.external import import_module
from sympy.external.gmpy import GROUND_TYPES, HAS_GMPY
from collections import namedtuple
def _timeout(self, function, timeout, fail_on_timeout):

    def callback(x, y):
        signal.alarm(0)
        if fail_on_timeout:
            raise TimeOutError('Timed out after %d seconds' % timeout)
        else:
            raise Skipped('Timeout')
    signal.signal(signal.SIGALRM, callback)
    signal.alarm(timeout)
    function()
    signal.alarm(0)