import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
class MockNumPy:
    """This is a mock numpy object that raises an error when there is an attempt
    to convert it to a boolean."""

    def __nonzero__(self):
        raise ValueError('The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()')