import os
import socket
import sys
import time
from breezy import config, controldir, errors, tests
from breezy import transport as _mod_transport
from breezy import ui
from breezy.osutils import lexists
from breezy.tests import TestCase, TestCaseWithTransport, TestSkipped, features
from breezy.tests.http_server import HttpServer
class ReadvFile:
    """An object that acts like Paramiko's SFTPFile when readv() is used"""

    def __init__(self, data):
        self._data = data

    def readv(self, requests):
        for start, length in requests:
            yield self._data[start:start + length]

    def close(self):
        pass