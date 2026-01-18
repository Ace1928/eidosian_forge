import atexit
import codecs
import contextlib
import copy
import difflib
import doctest
import errno
import functools
import itertools
import logging
import math
import os
import platform
import pprint
import random
import re
import shlex
import site
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import unittest
import warnings
from io import BytesIO, StringIO, TextIOWrapper
from typing import Callable, Set
import testtools
from testtools import content
import breezy
from breezy.bzr import chk_map
from .. import branchbuilder
from .. import commands as _mod_commands
from .. import config, controldir, debug, errors, hooks, i18n
from .. import lock as _mod_lock
from .. import lockdir, osutils
from .. import plugin as _mod_plugin
from .. import pyutils, registry, symbol_versioning, trace
from .. import transport as _mod_transport
from .. import ui, urlutils, workingtree
from ..bzr.smart import client, request
from ..tests import TestUtil, fixtures, test_server, treeshape, ui_testing
from ..transport import memory, pathfilter
from testtools.testcase import TestSkipped
class TextTestRunner:
    stop_on_failure = False

    def __init__(self, stream=sys.stderr, descriptions=0, verbosity=1, bench_history=None, strict=False, result_decorators=None):
        """Create a TextTestRunner.

        :param result_decorators: An optional list of decorators to apply
            to the result object being used by the runner. Decorators are
            applied left to right - the first element in the list is the
            innermost decorator.
        """
        new_encoding = osutils.get_terminal_encoding()
        codec = codecs.lookup(new_encoding)
        encode = codec.encode
        stream = osutils.UnicodeOrBytesToBytesWriter(encode, stream, 'backslashreplace')
        stream.encoding = new_encoding
        self.stream = stream
        self.descriptions = descriptions
        self.verbosity = verbosity
        self._bench_history = bench_history
        self._strict = strict
        self._result_decorators = result_decorators or []

    def run(self, test):
        """Run the given test case or test suite."""
        if self.verbosity == 1:
            result_class = TextTestResult
        elif self.verbosity >= 2:
            result_class = VerboseTestResult
        original_result = result_class(self.stream, self.descriptions, self.verbosity, bench_history=self._bench_history, strict=self._strict)
        original_result.stop_early = self.stop_on_failure
        result = original_result
        for decorator in self._result_decorators:
            result = decorator(result)
            result.stop_early = self.stop_on_failure
        result.startTestRun()
        try:
            test.run(result)
        finally:
            result.stopTestRun()
        return original_result