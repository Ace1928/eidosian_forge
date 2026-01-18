import copy
import functools
import itertools
import sys
import types
import unittest
import warnings
from testtools.compat import reraise
from testtools import content
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.matchers._basic import _FlippedEquals
from testtools.monkey import patch
from testtools.runtest import (
from testtools.testresult import (
def _unique_text(base_cp, cp_range, index):
    s = ''
    for m in _mods(index, cp_range):
        s += chr(base_cp + m)
    return s