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
class TestFailedSmartServerResponse(tests.TestCase):

    def test_construct(self):
        response = _mod_request.FailedSmartServerResponse((b'foo', b'bar'))
        self.assertEqual((b'foo', b'bar'), response.args)
        self.assertEqual(None, response.body)
        response = _mod_request.FailedSmartServerResponse((b'foo', b'bar'), b'bytes')
        self.assertEqual((b'foo', b'bar'), response.args)
        self.assertEqual(b'bytes', response.body)
        repr(response)

    def test_is_successful(self):
        """is_successful should return False for FailedSmartServerResponse."""
        response = _mod_request.FailedSmartServerResponse((b'error',))
        self.assertEqual(False, response.is_successful())