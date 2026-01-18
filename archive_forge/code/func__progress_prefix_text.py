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
def _progress_prefix_text(self):
    a = '[%d' % self.count
    if self.num_tests:
        a += '/%d' % self.num_tests
    a += ' in '
    runtime = time.time() - self._overall_start_time
    if runtime >= 60:
        a += '%dm%ds' % (runtime / 60, runtime % 60)
    else:
        a += '%ds' % runtime
    total_fail_count = self.error_count + self.failure_count
    if total_fail_count:
        a += ', %d failed' % total_fail_count
    a += ']'
    return a