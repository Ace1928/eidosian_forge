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
class mask_runner(object):

    def __init__(self, runner, mask, **options):
        self.runner = runner
        self.mask = mask

    def __call__(self):
        if self.mask:
            set_num_threads(self.mask)
        self.runner()