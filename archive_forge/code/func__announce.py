import unittest
import os
from test.support import captured_stdout
from distutils.cmd import Command
from distutils.dist import Distribution
from distutils.errors import DistutilsOptionError
from distutils import debug
def _announce(msg, level):
    msgs.append(msg)