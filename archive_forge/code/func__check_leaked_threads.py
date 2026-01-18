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
def _check_leaked_threads(self, test):
    """See if any threads have leaked since last call

        A sample of live threads is stored in the _active_threads attribute,
        when this method runs it compares the current live threads and any not
        in the previous sample are treated as having leaked.
        """
    now_active_threads = set(threading.enumerate())
    threads_leaked = now_active_threads.difference(self._active_threads)
    if threads_leaked:
        self._report_thread_leak(test, threads_leaked, now_active_threads)
        self._tests_leaking_threads_count += 1
        if self._first_thread_leaker_id is None:
            self._first_thread_leaker_id = test.id()
        self._active_threads = now_active_threads