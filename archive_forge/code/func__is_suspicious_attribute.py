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
def _is_suspicious_attribute(testCaseClass, name):
    """Returns True if an attribute is a method named like a test method."""
    if name.startswith('Test') and len(name) > 4 and name[4].isupper():
        attr = getattr(testCaseClass, name)
        if inspect.isfunction(attr) or inspect.ismethod(attr):
            args = inspect.getfullargspec(attr)
            return len(args.args) == 1 and args.args[0] == 'self' and (args.varargs is None) and (args.varkw is None) and (not args.kwonlyargs)
    return False