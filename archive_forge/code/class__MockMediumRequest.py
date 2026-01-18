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
class _MockMediumRequest:
    """A mock ClientMediumRequest used by MockMedium."""

    def __init__(self, mock_medium):
        self._medium = mock_medium
        self._written_bytes = b''
        self._read_bytes = b''
        self._response = None

    def accept_bytes(self, bytes):
        self._written_bytes += bytes

    def finished_writing(self):
        self._medium._assertEvent(('send request', self._written_bytes))
        self._written_bytes = b''

    def finished_reading(self):
        self._medium._assertEvent(('read response', self._read_bytes))
        self._read_bytes = b''

    def read_bytes(self, size):
        resp = self._response
        bytes, resp = (resp[:size], resp[size:])
        self._response = resp
        self._read_bytes += bytes
        return bytes

    def read_line(self):
        resp = self._response
        try:
            line, resp = resp.split(b'\n', 1)
            line += b'\n'
        except ValueError:
            line, resp = (resp, b'')
        self._response = resp
        self._read_bytes += line
        return line