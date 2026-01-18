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
class FakeNFSDecoratorTests(tests.TestCaseInTempDir):
    """NFS decorator specific tests."""

    def get_nfs_transport(self, url):
        return fakenfs.FakeNFSTransportDecorator('fakenfs+' + url)

    def test_local_parameters(self):
        t = self.get_nfs_transport('.')
        self.assertEqual(True, t.listable())
        self.assertEqual(False, t.is_readonly())

    def test_http_parameters(self):
        from breezy.tests.http_server import HttpServer
        server = HttpServer()
        self.start_server(server)
        t = self.get_nfs_transport(server.get_url())
        self.assertIsInstance(t, fakenfs.FakeNFSTransportDecorator)
        self.assertEqual(False, t.listable())
        self.assertEqual(True, t.is_readonly())

    def test_fakenfs_server_default(self):
        server = test_server.FakeNFSServer()
        self.start_server(server)
        self.assertStartsWith(server.get_url(), 'fakenfs+')
        t = transport.get_transport_from_url(server.get_url())
        self.assertIsInstance(t, fakenfs.FakeNFSTransportDecorator)

    def test_fakenfs_rename_semantics(self):
        t = self.get_nfs_transport('.')
        self.build_tree(['from/', 'from/foo', 'to/', 'to/bar'], transport=t)
        self.assertRaises(errors.ResourceBusy, t.rename, 'from', 'to')