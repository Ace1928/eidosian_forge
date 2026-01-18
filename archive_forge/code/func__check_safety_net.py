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
def _check_safety_net(self):
    """Check that the safety .bzr directory have not been touched.

        _make_test_root have created a .bzr directory to prevent tests from
        propagating. This method ensures than a test did not leaked.
        """
    root = TestCaseWithMemoryTransport.TEST_ROOT
    t = _mod_transport.get_transport_from_path(root)
    self.permit_url(t.base)
    if t.get_bytes('.bzr/checkout/dirstate') != TestCaseWithMemoryTransport._SAFETY_NET_PRISTINE_DIRSTATE:
        _rmtree_temp_dir(root + '/.bzr', test_id=self.id())
        self._create_safety_net()
        raise AssertionError('%s/.bzr should not be modified' % root)