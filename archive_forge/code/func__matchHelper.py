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
def _matchHelper(self, matchee, matcher, message, verbose):
    matcher = Annotate.if_message(message, matcher)
    mismatch = matcher.match(matchee)
    if not mismatch:
        return
    for name, value in mismatch.get_details().items():
        self.addDetailUniqueName(name, value)
    return MismatchError(matchee, matcher, mismatch, verbose)