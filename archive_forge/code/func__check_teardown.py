import atexit
import functools
import numpy
import os
import random
import types
import unittest
import cupy
@atexit.register
def _check_teardown():
    assert _nest_count == 0, '_setup_random() and _teardown_random() must be called in pairs.'