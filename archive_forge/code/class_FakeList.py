import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
class FakeList(list):

    def __getitem__(inner_self, i):
        self.fail('possibly side-effecting __getitem_ method called')