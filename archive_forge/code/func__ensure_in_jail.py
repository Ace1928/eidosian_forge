import doctest
import errno
import glob
import logging
import os
import shlex
import sys
import textwrap
from .. import osutils, tests, trace
from ..tests import ui_testing
def _ensure_in_jail(self, test_case, path):
    jail_root = self._get_jail_root(test_case)
    if not osutils.is_inside(jail_root, osutils.normalizepath(path)):
        raise ValueError('{} is not inside {}'.format(path, jail_root))