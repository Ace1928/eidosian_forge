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
def _capture_deprecation_warnings(self, a_callable, *args, **kwargs):
    """A helper for callDeprecated and applyDeprecated.

        :param a_callable: A callable to call.
        :param args: The positional arguments for the callable
        :param kwargs: The keyword arguments for the callable
        :return: A tuple (warnings, result). result is the result of calling
            a_callable(``*args``, ``**kwargs``).
        """
    local_warnings = []

    def capture_warnings(msg, cls=None, stacklevel=None):
        self.assertEqual(cls, DeprecationWarning)
        local_warnings.append(msg)
    original_warning_method = symbol_versioning.warn
    symbol_versioning.set_warning_method(capture_warnings)
    try:
        result = a_callable(*args, **kwargs)
    finally:
        symbol_versioning.set_warning_method(original_warning_method)
    return (local_warnings, result)