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
def _setup_filtering(argv: MutableSequence[str]) -> bool:
    """Implements the bazel test filtering protocol.

  The following environment variable is used in this method:

    TESTBRIDGE_TEST_ONLY: string, if set, is forwarded to the unittest
      framework to use as a test filter. Its value is split with shlex, then:
      1. On Python 3.6 and before, split values are passed as positional
         arguments on argv.
      2. On Python 3.7+, split values are passed to unittest's `-k` flag. Tests
         are matched by glob patterns or substring. See
         https://docs.python.org/3/library/unittest.html#cmdoption-unittest-k

  Args:
    argv: the argv to mutate in-place.

  Returns:
    Whether test filtering is requested.
  """
    test_filter = os.environ.get('TESTBRIDGE_TEST_ONLY')
    if argv is None or not test_filter:
        return False
    filters = shlex.split(test_filter)
    if sys.version_info[:2] >= (3, 7):
        filters = ['-k=' + test_filter for test_filter in filters]
    argv[1:1] = filters
    return True