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
def _clear__type_equality_funcs(test):
    """Cleanup bound methods stored on TestCase instances

    Clear the dict breaking a few (mostly) harmless cycles in the affected
    unittests released with Python 2.6 and initial Python 2.7 versions.

    For a few revisions between Python 2.7.1 and Python 2.7.2 that annoyingly
    shipped in Oneiric, an object with no clear method was used, hence the
    extra complications, see bug 809048 for details.
    """
    type_equality_funcs = getattr(test, '_type_equality_funcs', None)
    if type_equality_funcs is not None:
        tef_clear = getattr(type_equality_funcs, 'clear', None)
        if tef_clear is None:
            tef_instance_dict = getattr(type_equality_funcs, '__dict__', None)
            if tef_instance_dict is not None:
                tef_clear = tef_instance_dict.clear
        if tef_clear is not None:
            tef_clear()