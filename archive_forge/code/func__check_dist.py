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
def _check_dist(self, pop, samples):
    """
        Check distribution of some samples.
        """
    self.assertGreaterEqual(len(samples), len(pop) * 100)
    expected_frequency = len(samples) / len(pop)
    c = collections.Counter(samples)
    for value in pop:
        n = c[value]
        self.assertGreaterEqual(n, expected_frequency * 0.5)
        self.assertLessEqual(n, expected_frequency * 2.0)