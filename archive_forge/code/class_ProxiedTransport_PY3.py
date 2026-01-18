import io
import os
import re
import sys
import time
import socket
import base64
import tempfile
import logging
from pyomo.common.dependencies import attempt_import
class ProxiedTransport_PY3(xmlrpclib.Transport):

    def set_proxy(self, host):
        self.proxy = urlparse(host)
        if not self.proxy.hostname:
            self.proxy = urlparse('http://' + host)

    def make_connection(self, host):
        scheme = urlparse(host).scheme
        if not scheme:
            scheme = NEOS.scheme
        if scheme == 'https':
            connClass = httplib.HTTPSConnection
        else:
            connClass = httplib.HTTPConnection
        connection = connClass(self.proxy.hostname, self.proxy.port)
        connection.set_tunnel(host)
        return connection