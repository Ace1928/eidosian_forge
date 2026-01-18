import collections.abc
import copy
import errno
import functools
import http.client
import os
import re
import urllib.parse as urlparse
import osprofiler.web
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import netutils
from glance.common import auth
from glance.common import exception
from glance.common import utils
from glance.i18n import _
class HTTPSClientAuthConnection(http.client.HTTPSConnection):
    """
    Class to make a HTTPS connection, with support for
    full client-based SSL Authentication

    :see http://code.activestate.com/recipes/
            577548-https-httplib-client-connection-with-certificate-v/
    """

    def __init__(self, host, port, key_file, cert_file, ca_file, timeout=None, insecure=False):
        http.client.HTTPSConnection.__init__(self, host, port, key_file=key_file, cert_file=cert_file)
        self.key_file = key_file
        self.cert_file = cert_file
        self.ca_file = ca_file
        self.timeout = timeout
        self.insecure = insecure

    def connect(self):
        """
        Connect to a host on a given (SSL) port.
        If ca_file is pointing somewhere, use it to check Server Certificate.

        Redefined/copied and extended from httplib.py:1105 (Python 2.6.x).
        This is needed to pass cert_reqs=ssl.CERT_REQUIRED as parameter to
        ssl.wrap_socket(), which forces SSL to check server certificate against
        our client certificate.
        """
        sock = socket.create_connection((self.host, self.port), self.timeout)
        if self._tunnel_host:
            self.sock = sock
            self._tunnel()
        if self.insecure is True:
            self.sock = ssl.wrap_socket(sock, self.key_file, self.cert_file, cert_reqs=ssl.CERT_NONE)
        else:
            self.sock = ssl.wrap_socket(sock, self.key_file, self.cert_file, ca_certs=self.ca_file, cert_reqs=ssl.CERT_REQUIRED)