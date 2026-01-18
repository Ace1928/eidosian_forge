import doctest
import gc
import unittest as pyunit
from typing import Iterator, Union
from zope.interface import implementer
from twisted.python import components
from twisted.trial import itrial, reporter
from twisted.trial._synctest import _logObserver
def _clearSuite(suite):
    """
    Clear all tests from C{suite}.

    This messes with the internals of C{suite}. In particular, it assumes that
    the suite keeps all of its tests in a list in an instance variable called
    C{_tests}.
    """
    suite._tests = []