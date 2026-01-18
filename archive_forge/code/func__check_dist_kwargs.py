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
def _check_dist_kwargs(self, func, pyfunc, kwargslist, niters=3, prec='double', ulps=12, pydtype=None):
    assert len(kwargslist)
    for kwargs in kwargslist:
        results = [func(**kwargs) for i in range(niters)]
        pyresults = [pyfunc(**kwargs, dtype=pydtype) if pydtype else pyfunc(**kwargs) for i in range(niters)]
        self.assertPreciseEqual(results, pyresults, prec=prec, ulps=ulps, msg='for arguments %s' % (kwargs,))