import unittest
import inspect
from numba import njit
from numba.tests.support import TestCase
from numba.misc.firstlinefinder import get_func_body_first_lineno
def _get_grandparent_caller_code(self):
    frame = inspect.currentframe()
    caller_frame = inspect.getouterframes(frame)
    return caller_frame[2].frame.f_code