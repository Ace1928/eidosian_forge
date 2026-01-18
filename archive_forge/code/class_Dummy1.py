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
class Dummy1(object):

    def _repr_pretty_(self, p, cycle):
        p.text('Dummy1(...)')