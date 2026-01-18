import doctest
import gc
import unittest as pyunit
from typing import Iterator, Union
from zope.interface import implementer
from twisted.python import components
from twisted.trial import itrial, reporter
from twisted.trial._synctest import _logObserver
def _iterateTests(testSuiteOrCase: Union[pyunit.TestCase, pyunit.TestSuite]) -> Iterator[itrial.ITestCase]:
    """
    Iterate through all of the test cases in C{testSuiteOrCase}.
    """
    try:
        suite = iter(testSuiteOrCase)
    except TypeError:
        yield testSuiteOrCase
    else:
        for test in suite:
            yield from _iterateTests(test)