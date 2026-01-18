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
class FakeVFATDecoratorTests(tests.TestCaseInTempDir):
    """Tests for simulation of VFAT restrictions"""

    def get_vfat_transport(self, url):
        """Return vfat-backed transport for test directory"""
        from breezy.transport.fakevfat import FakeVFATTransportDecorator
        return FakeVFATTransportDecorator('vfat+' + url)

    def test_transport_creation(self):
        from breezy.transport.fakevfat import FakeVFATTransportDecorator
        t = self.get_vfat_transport('.')
        self.assertIsInstance(t, FakeVFATTransportDecorator)

    def test_transport_mkdir(self):
        t = self.get_vfat_transport('.')
        t.mkdir('HELLO')
        self.assertTrue(t.has('hello'))
        self.assertTrue(t.has('Hello'))

    def test_forbidden_chars(self):
        t = self.get_vfat_transport('.')
        self.assertRaises(ValueError, t.has, '<NU>')