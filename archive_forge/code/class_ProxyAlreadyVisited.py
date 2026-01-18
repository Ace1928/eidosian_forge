from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
class ProxyAlreadyVisited(object):
    """
    Placeholder proxy to use when protecting against infinite recursion due to
    loops in the object graph.

    Analogous to the values emitted by the users of Py_ReprEnter and Py_ReprLeave
    """

    def __init__(self, rep):
        self._rep = rep

    def __repr__(self):
        return self._rep