import doctest
import errno
import os
import socket
import subprocess
import sys
import threading
import time
from io import BytesIO
from typing import Optional, Type
from testtools.matchers import DocTestMatches
import breezy
from ... import controldir, debug, errors, osutils, tests
from ... import transport as _mod_transport
from ... import urlutils
from ...tests import features, test_server
from ...transport import local, memory, remote, ssh
from ...transport.http import urllib
from .. import bzrdir
from ..remote import UnknownErrorFromSmartServer
from ..smart import client, medium, message, protocol
from ..smart import request as _mod_request
from ..smart import server as _mod_server
from ..smart import vfs
from . import test_smart
class SmartServerRequestHandlerTests(tests.TestCaseWithTransport):
    """Test that call directly into the handler logic, bypassing the network."""

    def setUp(self):
        super().setUp()
        self.overrideEnv('BRZ_NO_SMART_VFS', None)

    def build_handler(self, transport):
        """Returns a handler for the commands in protocol version one."""
        return _mod_request.SmartServerRequestHandler(transport, _mod_request.request_handlers, '/')

    def test_construct_request_handler(self):
        """Constructing a request handler should be easy and set defaults."""
        handler = _mod_request.SmartServerRequestHandler(None, commands=None, root_client_path='/')
        self.assertFalse(handler.finished_reading)

    def test_hello(self):
        handler = self.build_handler(None)
        handler.args_received((b'hello',))
        self.assertEqual((b'ok', b'2'), handler.response.args)
        self.assertEqual(None, handler.response.body)

    def test_disable_vfs_handler_classes_via_environment(self):
        handler = vfs.HasRequest(None, '/')
        self.overrideEnv('BRZ_NO_SMART_VFS', '')
        self.assertRaises(_mod_request.DisabledMethod, handler.execute)

    def test_readonly_exception_becomes_transport_not_possible(self):
        """The response for a read-only error is ('ReadOnlyError')."""
        handler = self.build_handler(self.get_readonly_transport())
        handler.args_received((b'mkdir', b'foo', b''))
        self.assertEqual((b'ReadOnlyError',), handler.response.args)

    def test_hello_has_finished_body_on_dispatch(self):
        """The 'hello' command should set finished_reading."""
        handler = self.build_handler(None)
        handler.args_received((b'hello',))
        self.assertTrue(handler.finished_reading)
        self.assertNotEqual(None, handler.response)

    def test_put_bytes_non_atomic(self):
        """'put_...' should set finished_reading after reading the bytes."""
        handler = self.build_handler(self.get_transport())
        handler.args_received((b'put_non_atomic', b'a-file', b'', b'F', b''))
        self.assertFalse(handler.finished_reading)
        handler.accept_body(b'1234')
        self.assertFalse(handler.finished_reading)
        handler.accept_body(b'5678')
        handler.end_of_body()
        self.assertTrue(handler.finished_reading)
        self.assertEqual((b'ok',), handler.response.args)
        self.assertEqual(None, handler.response.body)

    def test_readv_accept_body(self):
        """'readv' should set finished_reading after reading offsets."""
        self.build_tree(['a-file'])
        handler = self.build_handler(self.get_readonly_transport())
        handler.args_received((b'readv', b'a-file'))
        self.assertFalse(handler.finished_reading)
        handler.accept_body(b'2,')
        self.assertFalse(handler.finished_reading)
        handler.accept_body(b'3')
        handler.end_of_body()
        self.assertTrue(handler.finished_reading)
        self.assertEqual((b'readv',), handler.response.args)
        self.assertEqual(b'nte', handler.response.body)

    def test_readv_short_read_response_contents(self):
        """'readv' when a short read occurs sets the response appropriately."""
        self.build_tree(['a-file'])
        handler = self.build_handler(self.get_readonly_transport())
        handler.args_received((b'readv', b'a-file'))
        handler.accept_body(b'100,1')
        handler.end_of_body()
        self.assertTrue(handler.finished_reading)
        self.assertEqual((b'ShortReadvError', b'./a-file', b'100', b'1', b'0'), handler.response.args)
        self.assertEqual(None, handler.response.body)