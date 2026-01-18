from __future__ import absolute_import
import os
import io
import sys
from contextlib import contextmanager
from unittest import skipIf
from Cython.Build import IpythonMagic
from Cython.TestUtils import CythonTest
from Cython.Compiler.Annotate import AnnotationCCodeWriter
from libc.math cimport sin
class MockLog:
    DEBUG = 1
    INFO = 2
    thresholds = [INFO]

    def set_threshold(self, val):
        self.thresholds.append(val)
        return self.thresholds[-2]