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
class TestTransportFromPath(tests.TestCaseInTempDir):

    def test_with_path(self):
        t = transport.get_transport_from_path(self.test_dir)
        self.assertIsInstance(t, local.LocalTransport)
        self.assertEqual(t.base.rstrip('/'), urlutils.local_path_to_url(self.test_dir))

    def test_with_url(self):
        t = transport.get_transport_from_path('file:')
        self.assertIsInstance(t, local.LocalTransport)
        self.assertEqual(t.base.rstrip('/'), urlutils.local_path_to_url(os.path.join(self.test_dir, 'file:')))