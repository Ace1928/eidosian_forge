import base64
import cgi
import errno
import http.client
import os
import re
import socket
import ssl
import sys
import time
import urllib
import urllib.request
import weakref
from urllib.parse import urlencode, urljoin, urlparse
from ... import __version__ as breezy_version
from ... import config, debug, errors, osutils, trace, transport, ui, urlutils
from ...bzr.smart import medium
from ...trace import mutter, mutter_callsite
from ...transport import ConnectedTransport, NoSuchFile, UnusableRedirect
from . import default_user_agent, ssl
from .response import handle_response
class _ReportingFileSocket:

    def __init__(self, filesock, report_activity=None):
        self.filesock = filesock
        self._report_activity = report_activity

    def report_activity(self, size, direction):
        if self._report_activity:
            self._report_activity(size, direction)

    def read(self, size=1):
        s = self.filesock.read(size)
        self.report_activity(len(s), 'read')
        return s

    def readline(self, size=-1):
        s = self.filesock.readline(size)
        self.report_activity(len(s), 'read')
        return s

    def readinto(self, b):
        s = self.filesock.readinto(b)
        self.report_activity(s, 'read')
        return s

    def __getattr__(self, name):
        return getattr(self.filesock, name)