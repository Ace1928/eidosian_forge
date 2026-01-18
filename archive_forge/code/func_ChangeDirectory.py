from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import contextlib
import os
import re
import sys
import unittest
from fire import core
from fire import trace
import mock
import six
@contextlib.contextmanager
def ChangeDirectory(directory):
    """Context manager to mock a directory change and revert on exit."""
    cwdir = os.getcwd()
    os.chdir(directory)
    try:
        yield directory
    finally:
        os.chdir(cwdir)