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
def _check_any_distrib_kwargs(self, func, ptr, distrib, paramlist):
    """
        Check any numpy distribution function. Does Numba use the same keyword
        argument names as Numpy?
        And given a fixed seed, do they both return the same samples?
        """
    r = self._follow_numpy(ptr)
    distrib_method_of_numpy = getattr(r, distrib)
    self._check_dist_kwargs(func, distrib_method_of_numpy, paramlist)