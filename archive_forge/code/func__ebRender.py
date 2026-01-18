import base64
import xmlrpc.client as xmlrpclib
from urllib.parse import urlparse
from xmlrpc.client import Binary, Boolean, DateTime, Fault
from twisted.internet import defer, error, protocol
from twisted.logger import Logger
from twisted.python import failure, reflect
from twisted.python.compat import nativeString
from twisted.web import http, resource, server
def _ebRender(self, failure):
    if isinstance(failure.value, Fault):
        return failure.value
    self._log.failure('', failure)
    return Fault(self.FAILURE, 'error')