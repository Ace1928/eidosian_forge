import sys
from unittest import mock
import types
import warnings
import unittest
import os
import subprocess
import threading
from numba import config, njit
from numba.tests.support import TestCase
from numba.testing.main import _TIMEOUT as _RUNNER_TIMEOUT
class _DummyClass(object):

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return '_DummyClass(%f, %f)' % self.value