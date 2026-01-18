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
class TestGetProtocolFactoryForBytes(tests.TestCase):
    """_get_protocol_factory_for_bytes identifies the protocol factory a server
    should use to decode a given request.  Any bytes not part of the version
    marker string (and thus part of the actual request) are returned alongside
    the protocol factory.
    """

    def test_version_three(self):
        result = medium._get_protocol_factory_for_bytes(b'bzr message 3 (bzr 1.6)\nextra bytes')
        protocol_factory, remainder = result
        self.assertEqual(protocol.build_server_protocol_three, protocol_factory)
        self.assertEqual(b'extra bytes', remainder)

    def test_version_two(self):
        result = medium._get_protocol_factory_for_bytes(b'bzr request 2\nextra bytes')
        protocol_factory, remainder = result
        self.assertEqual(protocol.SmartServerRequestProtocolTwo, protocol_factory)
        self.assertEqual(b'extra bytes', remainder)

    def test_version_one(self):
        """Version one requests have no version markers."""
        result = medium._get_protocol_factory_for_bytes(b'anything\n')
        protocol_factory, remainder = result
        self.assertEqual(protocol.SmartServerRequestProtocolOne, protocol_factory)
        self.assertEqual(b'anything\n', remainder)