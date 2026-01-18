import errno
import os
import subprocess
import sys
import threading
from io import BytesIO
import breezy.transport.trace
from .. import errors, osutils, tests, transport, urlutils
from ..transport import (FileExists, NoSuchFile, UnsupportedProtocol, chroot,
from . import features, test_server
class TestWin32LocalTransport(tests.TestCase):

    def test_unc_clone_to_root(self):
        self.requireFeature(features.win32_feature)
        t = local.EmulatedWin32LocalTransport('file://HOST/path/to/some/dir/')
        for i in range(4):
            t = t.clone('..')
        self.assertEqual(t.base, 'file://HOST/')
        t = t.clone('..')
        self.assertEqual(t.base, 'file://HOST/')