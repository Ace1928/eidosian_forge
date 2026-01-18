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
class _method(object):
    """A decorator that supports both instance and classmethod invocations.

  Using similar semantics to the @property builtin, this decorator can augment
  an instance method to support conditional logic when invoked on a class
  object. This breaks support for invoking an instance method via the class
  (e.g. Cls.method(self, ...)) but is still situationally useful.
  """

    def __init__(self, finstancemethod):
        self._finstancemethod = finstancemethod
        self._fclassmethod = None

    def classmethod(self, fclassmethod):
        self._fclassmethod = classmethod(fclassmethod)
        return self

    def __doc__(self):
        if getattr(self._finstancemethod, '__doc__'):
            return self._finstancemethod.__doc__
        elif getattr(self._fclassmethod, '__doc__'):
            return self._fclassmethod.__doc__
        return ''

    def __get__(self, obj, type_):
        func = self._fclassmethod if obj is None else self._finstancemethod
        return func.__get__(obj, type_)