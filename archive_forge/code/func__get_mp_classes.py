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
def _get_mp_classes(method):
    if method == 'default':
        method = None
    ctx = multiprocessing.get_context(method)
    proc = _proc_class_impl(method)
    queue = ctx.Queue
    return (proc, queue)