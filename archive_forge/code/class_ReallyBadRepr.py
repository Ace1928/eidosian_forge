from collections import Counter, defaultdict, deque, OrderedDict, UserList
import os
import pytest
import types
import string
import sys
import unittest
import pytest
from IPython.lib import pretty
from io import StringIO
class ReallyBadRepr(object):
    __module__ = 1

    @property
    def __class__(self):
        raise ValueError('I am horrible')

    def __repr__(self):
        raise BadException()