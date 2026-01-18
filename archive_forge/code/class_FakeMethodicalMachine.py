from __future__ import print_function
import functools
import os
import subprocess
from unittest import TestCase, skipIf
import attr
from .._methodical import MethodicalMachine
from .test_discover import isTwistedInstalled
class FakeMethodicalMachine(object):
    """
    A fake L{MethodicalMachine}.  Instantiate it with a L{FakeDigraph}
    """

    def __init__(self, digraph):
        self._digraph = digraph

    def asDigraph(self):
        return self._digraph