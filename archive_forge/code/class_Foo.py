import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
class Foo:
    a = 10

    def __init__(self):
        self.b = 20

    def method(self, x):
        pass