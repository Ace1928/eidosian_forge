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
def _rmtree_temp_dir(dirname, test_id=None):
    if sys.platform == 'win32' and isinstance(dirname, bytes):
        dirname = dirname.decode('mbcs')
    else:
        dirname = dirname.encode(sys.getfilesystemencoding())
    try:
        osutils.rmtree(dirname)
    except OSError as e:
        if test_id is not None:
            ui.ui_factory.clear_term()
            sys.stderr.write('\nWhile running: {}\n'.format(test_id))
        printable_e = str(e).decode(osutils.get_user_encoding(), 'replace').encode('ascii', 'replace')
        sys.stderr.write('Unable to remove testing dir %s\n%s' % (os.path.basename(dirname), printable_e))