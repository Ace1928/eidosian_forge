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
def _quote_long_string(s):
    """Quotes a potentially multi-line string to make the start and end obvious.

  Args:
    s: A string.

  Returns:
    The quoted string.
  """
    if isinstance(s, (bytes, bytearray)):
        try:
            s = s.decode('utf-8')
        except UnicodeDecodeError:
            s = str(s)
    return '8<-----------\n' + s + '\n' + '----------->8\n'