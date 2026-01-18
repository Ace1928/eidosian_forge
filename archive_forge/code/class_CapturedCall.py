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
class CapturedCall:
    """A helper for capturing smart server calls for easy debug analysis."""

    def __init__(self, params, prefix_length):
        """Capture the call with params and skip prefix_length stack frames."""
        self.call = params
        import traceback
        stack = traceback.extract_stack()[prefix_length:-5]
        self._stack = ''.join(traceback.format_list(stack))

    def __str__(self):
        return self.call.method.decode('utf-8')

    def __repr__(self):
        return self.call.method.decode('utf-8')

    def stack(self):
        return self._stack