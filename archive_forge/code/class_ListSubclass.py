import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
class ListSubclass(list):
    pass