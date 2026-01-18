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
class LoggingMessageHandler:

    def __init__(self):
        self.event_log = []

    def _log(self, *args):
        self.event_log.append(args)

    def headers_received(self, headers):
        self._log('headers', headers)

    def protocol_error(self, exception):
        self._log('protocol_error', exception)

    def byte_part_received(self, byte):
        self._log('byte', byte)

    def bytes_part_received(self, bytes):
        self._log('bytes', bytes)

    def structure_part_received(self, structure):
        self._log('structure', structure)

    def end_received(self):
        self._log('end')