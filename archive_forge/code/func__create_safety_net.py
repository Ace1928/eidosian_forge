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
def _create_safety_net(self):
    """Make a fake bzr directory.

        This prevents any tests propagating up onto the TEST_ROOT directory's
        real branch.
        """
    root = TestCaseWithMemoryTransport.TEST_ROOT
    try:
        self.assertIs(None, os.environ.get('BRZ_HOME', None))
        os.environ['BRZ_HOME'] = root
        from breezy.bzr.bzrdir import BzrDirMetaFormat1
        wt = controldir.ControlDir.create_standalone_workingtree(root, format=BzrDirMetaFormat1())
        del os.environ['BRZ_HOME']
    except Exception as e:
        self.fail('Fail to initialize the safety net: {!r}\n'.format(e))
    TestCaseWithMemoryTransport._SAFETY_NET_PRISTINE_DIRSTATE = wt.control_transport.get_bytes('dirstate')