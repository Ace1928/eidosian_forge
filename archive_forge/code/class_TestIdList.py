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
class TestIdList:
    """Test id list to filter a test suite.

    Relying on the assumption that test ids are built as:
    <module>[.<class>.<method>][(<param>+)], <module> being in python dotted
    notation, this class offers methods to :
    - avoid building a test suite for modules not refered to in the test list,
    - keep only the tests listed from the module test suite.
    """

    def __init__(self, test_id_list):
        self.tests = dict().fromkeys(test_id_list, True)
        modules = {}
        for test_id in test_id_list:
            parts = test_id.split('.')
            mod_name = parts.pop(0)
            modules[mod_name] = True
            for part in parts:
                mod_name += '.' + part
                modules[mod_name] = True
        self.modules = modules

    def refers_to(self, module_name):
        """Is there tests for the module or one of its sub modules."""
        return module_name in self.modules

    def includes(self, test_id):
        return test_id in self.tests