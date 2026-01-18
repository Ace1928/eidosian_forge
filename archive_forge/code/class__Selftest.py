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
class _Selftest:
    """Mixin for tests needing full selftest output"""

    def _inject_stream_into_subunit(self, stream):
        """To be overridden by subclasses that run tests out of process"""

    def _run_selftest(self, **kwargs):
        bio = BytesIO()
        sio = TextIOWrapper(bio, 'utf-8')
        self._inject_stream_into_subunit(bio)
        tests.selftest(stream=sio, stop_on_failure=False, **kwargs)
        sio.flush()
        return bio.getvalue()