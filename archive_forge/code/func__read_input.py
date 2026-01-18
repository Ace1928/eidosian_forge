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
def _read_input(self, input, in_name):
    if in_name is not None:
        infile = open(in_name)
        try:
            input = infile.read()
        finally:
            infile.close()
    return input