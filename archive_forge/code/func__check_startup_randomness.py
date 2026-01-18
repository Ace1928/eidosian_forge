import collections
import functools
import math
import multiprocessing
import os
import random
import subprocess
import sys
import threading
import itertools
from textwrap import dedent
import numpy as np
import unittest
import numba
from numba import jit, _helperlib, njit
from numba.core import types
from numba.tests.support import TestCase, compile_function, tag
from numba.core.errors import TypingError
def _check_startup_randomness(self, func_name, func_args):
    """
        Check that the state is properly randomized at startup.
        """
    code = 'if 1:\n            from numba.tests import test_random\n            func = getattr(test_random, %(func_name)r)\n            print(func(*%(func_args)r))\n            ' % locals()
    numbers = set()
    for i in range(3):
        popen = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = popen.communicate()
        if popen.returncode != 0:
            raise AssertionError('process failed with code %s: stderr follows\n%s\n' % (popen.returncode, err.decode()))
        numbers.add(float(out.strip()))
    self.assertEqual(len(numbers), 3, numbers)