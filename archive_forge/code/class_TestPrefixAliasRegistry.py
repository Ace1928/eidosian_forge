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
class TestPrefixAliasRegistry(registry.Registry):
    """A registry for test prefix aliases.

    This helps implement shorcuts for the --starting-with selftest
    option. Overriding existing prefixes is not allowed but not fatal (a
    warning will be emitted).
    """

    def register(self, key, obj, help=None, info=None, override_existing=False):
        """See Registry.register.

        Trying to override an existing alias causes a warning to be emitted,
        not a fatal execption.
        """
        try:
            super().register(key, obj, help=help, info=info, override_existing=False)
        except KeyError:
            actual = self.get(key)
            trace.note('Test prefix alias %s is already used for %s, ignoring %s' % (key, actual, obj))

    def resolve_alias(self, id_start):
        """Replace the alias by the prefix in the given string.

        Using an unknown prefix is an error to help catching typos.
        """
        parts = id_start.split('.')
        try:
            parts[0] = self.get(parts[0])
        except KeyError:
            raise errors.CommandError('%s is not a known test prefix alias' % parts[0])
        return '.'.join(parts)