from collections import abc
import contextlib
import dataclasses
import difflib
import enum
import errno
import faulthandler
import getpass
import inspect
import io
import itertools
import json
import os
import random
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import typing
from typing import Any, AnyStr, BinaryIO, Callable, ContextManager, IO, Iterator, List, Mapping, MutableMapping, MutableSequence, NoReturn, Optional, Sequence, Text, TextIO, Tuple, Type, Union
import unittest
from unittest import mock  # pylint: disable=unused-import Allow absltest.mock.
from urllib import parse
from absl import app  # pylint: disable=g-import-not-at-top
from absl import flags
from absl import logging
from absl.testing import _pretty_print_reporter
from absl.testing import xml_reporter
def _monkey_patch_test_result_for_unexpected_passes():
    """Workaround for <http://bugs.python.org/issue20165>."""

    def wasSuccessful(self):
        """Tells whether or not this result was a success.

    Any unexpected pass is to be counted as a non-success.

    Args:
      self: The TestResult instance.

    Returns:
      Whether or not this result was a success.
    """
        return len(self.failures) == len(self.errors) == len(self.unexpectedSuccesses) == 0
    test_result = unittest.TestResult()
    test_result.addUnexpectedSuccess(unittest.FunctionTestCase(lambda: None))
    if test_result.wasSuccessful():
        unittest.TestResult.wasSuccessful = wasSuccessful
        if test_result.wasSuccessful():
            sys.stderr.write('unittest.result.TestResult monkey patch to report unexpected passes as failures did not work.\n')