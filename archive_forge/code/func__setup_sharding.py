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
def _setup_sharding(custom_loader: Optional[unittest.TestLoader]=None) -> Tuple[unittest.TestLoader, Optional[int]]:
    """Implements the bazel sharding protocol.

  The following environment variables are used in this method:

    TEST_SHARD_STATUS_FILE: string, if set, points to a file. We write a blank
      file to tell the test runner that this test implements the test sharding
      protocol.

    TEST_TOTAL_SHARDS: int, if set, sharding is requested.

    TEST_SHARD_INDEX: int, must be set if TEST_TOTAL_SHARDS is set. Specifies
      the shard index for this instance of the test process. Must satisfy:
      0 <= TEST_SHARD_INDEX < TEST_TOTAL_SHARDS.

  Args:
    custom_loader: A TestLoader to be made sharded.

  Returns:
    A tuple of ``(test_loader, shard_index)``. ``test_loader`` is for
    shard-filtering or the standard test loader depending on the sharding
    environment variables. ``shard_index`` is the shard index, or ``None`` when
    sharding is not used.
  """
    if 'TEST_SHARD_STATUS_FILE' in os.environ:
        try:
            with open(os.environ['TEST_SHARD_STATUS_FILE'], 'w') as f:
                f.write('')
        except IOError:
            sys.stderr.write('Error opening TEST_SHARD_STATUS_FILE (%s). Exiting.' % os.environ['TEST_SHARD_STATUS_FILE'])
            sys.exit(1)
    base_loader = custom_loader or TestLoader()
    if 'TEST_TOTAL_SHARDS' not in os.environ:
        return (base_loader, None)
    total_shards = int(os.environ['TEST_TOTAL_SHARDS'])
    shard_index = int(os.environ['TEST_SHARD_INDEX'])
    if shard_index < 0 or shard_index >= total_shards:
        sys.stderr.write('ERROR: Bad sharding values. index=%d, total=%d\n' % (shard_index, total_shards))
        sys.exit(1)
    delegate_get_names = base_loader.getTestCaseNames
    bucket_iterator = itertools.cycle(range(total_shards))

    def getShardedTestCaseNames(testCaseClass):
        filtered_names = []
        ordered_names = delegate_get_names(testCaseClass)
        for testcase in sorted(ordered_names):
            bucket = next(bucket_iterator)
            if bucket == shard_index:
                filtered_names.append(testcase)
        return [x for x in ordered_names if x in filtered_names]
    base_loader.getTestCaseNames = getShardedTestCaseNames
    return (base_loader, shard_index)