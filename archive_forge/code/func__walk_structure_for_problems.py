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
def _walk_structure_for_problems(a, b, aname, bname, problem_list, leaf_assert_equal_func, failure_exception):
    """The recursive comparison behind assertSameStructure."""
    if type(a) != type(b) and (not (_are_both_of_integer_type(a, b) or _are_both_of_sequence_type(a, b) or _are_both_of_set_type(a, b) or _are_both_of_mapping_type(a, b))):
        problem_list.append('%s is a %r but %s is a %r' % (aname, type(a), bname, type(b)))
        return
    if isinstance(a, abc.Set):
        for k in a:
            if k not in b:
                problem_list.append('%s has %r but %s does not' % (aname, k, bname))
        for k in b:
            if k not in a:
                problem_list.append('%s lacks %r but %s has it' % (aname, k, bname))
    elif isinstance(a, abc.Mapping):
        for k in a:
            if k in b:
                _walk_structure_for_problems(a[k], b[k], '%s[%r]' % (aname, k), '%s[%r]' % (bname, k), problem_list, leaf_assert_equal_func, failure_exception)
            else:
                problem_list.append("%s has [%r] with value %r but it's missing in %s" % (aname, k, a[k], bname))
        for k in b:
            if k not in a:
                problem_list.append('%s lacks [%r] but %s has it with value %r' % (aname, k, bname, b[k]))
    elif isinstance(a, abc.Sequence) and (not isinstance(a, _TEXT_OR_BINARY_TYPES)):
        minlen = min(len(a), len(b))
        for i in range(minlen):
            _walk_structure_for_problems(a[i], b[i], '%s[%d]' % (aname, i), '%s[%d]' % (bname, i), problem_list, leaf_assert_equal_func, failure_exception)
        for i in range(minlen, len(a)):
            problem_list.append('%s has [%i] with value %r but %s does not' % (aname, i, a[i], bname))
        for i in range(minlen, len(b)):
            problem_list.append('%s lacks [%i] but %s has it with value %r' % (aname, i, bname, b[i]))
    else:
        try:
            leaf_assert_equal_func(a, b)
        except failure_exception:
            problem_list.append('%s is %r but %s is %r' % (aname, a, bname, b))