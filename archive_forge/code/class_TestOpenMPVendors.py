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
@skip_parfors_unsupported
@skip_no_omp
class TestOpenMPVendors(TestCase):

    def test_vendors(self):
        """
        Checks the OpenMP vendor strings are correct
        """
        expected = dict()
        expected['win32'] = 'MS'
        expected['darwin'] = 'Intel'
        expected['linux'] = 'GNU'
        for k in expected.keys():
            if sys.platform.startswith(k):
                self.assertEqual(expected[k], omppool.openmp_vendor)