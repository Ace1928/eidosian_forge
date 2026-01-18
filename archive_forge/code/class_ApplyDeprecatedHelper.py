import doctest
import gc
import os
import signal
import sys
import threading
import time
import unittest
import warnings
from functools import reduce
from io import BytesIO, StringIO, TextIOWrapper
import testtools.testresult.doubles
from testtools import ExtendedToOriginalDecorator, MultiTestResult
from testtools.content import Content
from testtools.content_type import ContentType
from testtools.matchers import DocTestMatches, Equals
import breezy
from .. import (branchbuilder, controldir, errors, hooks, lockdir, memorytree,
from ..bzr import (bzrdir, groupcompress_repo, remote, workingtree_3,
from ..git import workingtree as git_workingtree
from ..symbol_versioning import (deprecated_function, deprecated_in,
from ..trace import mutter, note
from ..transport import memory
from . import TestUtil, features, test_lsprof, test_server
class ApplyDeprecatedHelper:
    """A helper class for ApplyDeprecated tests."""

    @deprecated_method(deprecated_in((0, 11, 0)))
    def sample_deprecated_method(self, param_one):
        """A deprecated method for testing with."""
        return param_one

    def sample_normal_method(self):
        """A undeprecated method."""

    @deprecated_method(deprecated_in((0, 10, 0)))
    def sample_nested_deprecation(self):
        return sample_deprecated_function()