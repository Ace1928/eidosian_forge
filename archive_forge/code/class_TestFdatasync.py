import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
class TestFdatasync(tests.TestCaseInTempDir):

    def do_fdatasync(self):
        f = tempfile.NamedTemporaryFile()
        osutils.fdatasync(f.fileno())
        f.close()

    @staticmethod
    def raise_eopnotsupp(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, os.strerror(errno.EOPNOTSUPP))

    @staticmethod
    def raise_enotsup(*args, **kwargs):
        raise OSError(errno.ENOTSUP, os.strerror(errno.ENOTSUP))

    def test_fdatasync_handles_system_function(self):
        self.overrideAttr(os, 'fdatasync')
        self.do_fdatasync()

    def test_fdatasync_handles_no_fdatasync_no_fsync(self):
        self.overrideAttr(os, 'fdatasync')
        self.overrideAttr(os, 'fsync')
        self.do_fdatasync()

    def test_fdatasync_handles_no_EOPNOTSUPP(self):
        self.overrideAttr(errno, 'EOPNOTSUPP')
        self.do_fdatasync()

    def test_fdatasync_catches_ENOTSUP(self):
        enotsup = getattr(errno, 'ENOTSUP', None)
        if enotsup is None:
            raise tests.TestNotApplicable('No ENOTSUP on this platform')
        self.overrideAttr(os, 'fdatasync', self.raise_enotsup)
        self.do_fdatasync()

    def test_fdatasync_catches_EOPNOTSUPP(self):
        enotsup = getattr(errno, 'EOPNOTSUPP', None)
        if enotsup is None:
            raise tests.TestNotApplicable('No EOPNOTSUPP on this platform')
        self.overrideAttr(os, 'fdatasync', self.raise_eopnotsupp)
        self.do_fdatasync()