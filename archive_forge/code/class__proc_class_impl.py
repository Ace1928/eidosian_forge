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
class _proc_class_impl(object):

    def __init__(self, method):
        self._method = method

    def __call__(self, *args, **kwargs):
        ctx = multiprocessing.get_context(self._method)
        return ctx.Process(*args, **kwargs)