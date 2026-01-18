import os
import re
import sys
import traceback
import types
import functools
import warnings
from fnmatch import fnmatch, fnmatchcase
from . import case, suite, util
def _make_skipped_test(methodname, exception, suiteClass):

    @case.skip(str(exception))
    def testSkipped(self):
        pass
    attrs = {methodname: testSkipped}
    TestClass = type('ModuleSkipped', (case.TestCase,), attrs)
    return suiteClass((TestClass(methodname),))