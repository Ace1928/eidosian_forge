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
def _setup_test_runner_fail_fast(argv):
    """Implements the bazel test fail fast protocol.

  The following environment variable is used in this method:

    TESTBRIDGE_TEST_RUNNER_FAIL_FAST=<1|0>

  If set to 1, --failfast is passed to the unittest framework to return upon
  first failure.

  Args:
    argv: the argv to mutate in-place.
  """
    if argv is None:
        return
    if os.environ.get('TESTBRIDGE_TEST_RUNNER_FAIL_FAST') != '1':
        return
    argv[1:1] = ['--failfast']