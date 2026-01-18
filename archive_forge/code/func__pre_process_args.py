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
def _pre_process_args(self, args):
    new_args = []
    for arg in args:
        if arg[0] in ('"', "'") and arg[0] == arg[-1]:
            yield arg[1:-1]
        elif glob.has_magic(arg):
            matches = glob.glob(arg)
            if matches:
                matches.sort()
                yield from matches
        else:
            yield arg