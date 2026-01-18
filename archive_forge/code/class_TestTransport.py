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
class TestTransport(tests.TestCase):
    """Test the non transport-concrete class functionality."""

    def test__get_set_protocol_handlers(self):
        handlers = transport._get_protocol_handlers()
        self.assertNotEqual([], handlers.keys())
        transport._clear_protocol_handlers()
        self.addCleanup(transport._set_protocol_handlers, handlers)
        self.assertEqual([], transport._get_protocol_handlers().keys())

    def test_get_transport_modules(self):
        handlers = transport._get_protocol_handlers()
        self.addCleanup(transport._set_protocol_handlers, handlers)
        transport._clear_protocol_handlers()

        class SampleHandler:
            """I exist, isnt that enough?"""
        transport._clear_protocol_handlers()
        transport.register_transport_proto('foo')
        transport.register_lazy_transport('foo', 'breezy.tests.test_transport', 'TestTransport.SampleHandler')
        transport.register_transport_proto('bar')
        transport.register_lazy_transport('bar', 'breezy.tests.test_transport', 'TestTransport.SampleHandler')
        self.assertEqual([SampleHandler.__module__, 'breezy.transport.chroot', 'breezy.transport.pathfilter'], transport._get_transport_modules())

    def test_transport_dependency(self):
        """Transport with missing dependency causes no error"""
        saved_handlers = transport._get_protocol_handlers()
        self.addCleanup(transport._set_protocol_handlers, saved_handlers)
        transport._clear_protocol_handlers()
        transport.register_transport_proto('foo')
        transport.register_lazy_transport('foo', 'breezy.tests.test_transport', 'BadTransportHandler')
        try:
            transport.get_transport_from_url('foo://fooserver/foo')
        except UnsupportedProtocol as e:
            self.assertEqual('Unsupported protocol for url "foo://fooserver/foo": Unable to import library "some_lib": testing missing dependency', str(e))
        else:
            self.fail('Did not raise UnsupportedProtocol')

    def test_transport_fallback(self):
        """Transport with missing dependency causes no error"""
        saved_handlers = transport._get_protocol_handlers()
        self.addCleanup(transport._set_protocol_handlers, saved_handlers)
        transport._clear_protocol_handlers()
        transport.register_transport_proto('foo')
        transport.register_lazy_transport('foo', 'breezy.tests.test_transport', 'BackupTransportHandler')
        transport.register_lazy_transport('foo', 'breezy.tests.test_transport', 'BadTransportHandler')
        t = transport.get_transport_from_url('foo://fooserver/foo')
        self.assertTrue(isinstance(t, BackupTransportHandler))

    def test_ssh_hints(self):
        """Transport ssh:// should raise an error pointing out bzr+ssh://"""
        try:
            transport.get_transport_from_url('ssh://fooserver/foo')
        except UnsupportedProtocol as e:
            self.assertEqual('Unsupported protocol for url "ssh://fooserver/foo": Use bzr+ssh for Bazaar operations over SSH, e.g. "bzr+ssh://fooserver/foo". Use git+ssh for Git operations over SSH, e.g. "git+ssh://fooserver/foo".', str(e))
        else:
            self.fail('Did not raise UnsupportedProtocol')

    def test_LateReadError(self):
        """The LateReadError helper should raise on read()."""
        a_file = transport.LateReadError('a path')
        try:
            a_file.read()
        except errors.ReadError as error:
            self.assertEqual('a path', error.path)
        self.assertRaises(errors.ReadError, a_file.read, 40)
        a_file.close()

    def test_local_abspath_non_local_transport(self):
        t = memory.MemoryTransport()
        e = self.assertRaises(errors.NotLocalUrl, t.local_abspath, 't')
        self.assertEqual('memory:///t is not a local path.', str(e))