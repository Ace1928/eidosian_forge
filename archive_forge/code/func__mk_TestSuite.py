import unittest
import gc
import sys
import weakref
from unittest.test.support import LoggingResult, TestEquality
def _mk_TestSuite(*names):
    return unittest.TestSuite((Test.Foo(n) for n in names))