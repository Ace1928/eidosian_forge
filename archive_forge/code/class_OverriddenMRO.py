import inspect
import os
import sys
import unittest
from collections.abc import Sequence
from typing import List
from bpython import inspection
from bpython.test.fodder import encoding_ascii
from bpython.test.fodder import encoding_latin1
from bpython.test.fodder import encoding_utf8
class OverriddenMRO:

    def __mro__(self):
        raise AssertionError('custom mro executed')
    a = 1