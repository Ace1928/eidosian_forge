import os
import io
import sys
import unittest
import warnings
import textwrap
from unittest import mock
from distutils.dist import Distribution, fix_help_options
from distutils.cmd import Command
from test.support import (
from test.support.os_helper import TESTFN
from distutils.tests import support
from distutils import log
class TestDistribution(Distribution):
    """Distribution subclasses that avoids the default search for
    configuration files.

    The ._config_files attribute must be set before
    .parse_config_files() is called.
    """

    def find_config_files(self):
        return self._config_files