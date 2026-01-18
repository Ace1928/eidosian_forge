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
def _mods(i, mod):
    q, r = divmod(i, mod)
    while True:
        yield r
        if not q:
            break
        q, r = divmod(q, mod)