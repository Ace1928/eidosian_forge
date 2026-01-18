import faulthandler
import itertools
import multiprocessing
import os
import random
import re
import subprocess
import sys
import textwrap
import threading
import unittest
import numpy as np
from numba import jit, vectorize, guvectorize, set_num_threads
from numba.tests.support import (temp_directory, override_config, TestCase, tag,
import queue as t_queue
from numba.testing.main import _TIMEOUT as _RUNNER_TIMEOUT
from numba.core import config
class linalg_runner(runnable):

    def __call__(self):
        cfunc = jit(**self._options)(linalg)
        a = 4
        b = 10
        expected = linalg(a, b)
        got = cfunc(a, b)
        np.testing.assert_allclose(expected, got)