import base64
import xmlrpc.client as xmlrpclib
from urllib.parse import urlparse
from xmlrpc.client import Binary, Boolean, DateTime, Fault
from twisted.internet import defer, error, protocol
from twisted.logger import Logger
from twisted.python import failure, reflect
from twisted.python.compat import nativeString
from twisted.web import http, resource, server
class QueryFactory(protocol.ClientFactory):
    """
    XML-RPC Client Factory

    @ivar path: The path portion of the URL to which to post method calls.
    @type path: L{bytes}

    @ivar host: The value to use for the Host HTTP header.
    @type host: L{bytes}

    @ivar user: The username with which to authenticate with the server
        when making calls.
    @type user: L{bytes} or L{None}

    @ivar password: The password with which to authenticate with the server
        when making calls.
    @type password: L{bytes} or L{None}

    @ivar useDateTime: Accept datetime values as datetime.datetime objects.
        also passed to the underlying xmlrpclib implementation.  Defaults to
        C{False}.
    @type useDateTime: C{bool}
    """
    deferred = None
    protocol = QueryProtocol

    def __init__(self, path, host, method, user=None, password=None, allowNone=False, args=(), canceller=None, useDateTime=False):
        """
        @param method: The name of the method to call.
        @type method: C{str}

        @param allowNone: allow the use of None values in parameters. It's
            passed to the underlying xmlrpclib implementation. Defaults to
            C{False}.
        @type allowNone: C{bool} or L{None}

        @param args: the arguments to pass to the method.
        @type args: C{tuple}

        @param canceller: A 1-argument callable passed to the deferred as the
            canceller callback.
        @type canceller: callable or L{None}
        """
        self.path, self.host = (path, host)
        self.user, self.password = (user, password)
        self.payload = payloadTemplate % (method, xmlrpclib.dumps(args, allow_none=allowNone))
        if isinstance(self.payload, str):
            self.payload = self.payload.encode('utf8')
        self.deferred = defer.Deferred(canceller)
        self.useDateTime = useDateTime

    def parseResponse(self, contents):
        if not self.deferred:
            return
        try:
            response = xmlrpclib.loads(contents, use_datetime=self.useDateTime)[0][0]
        except BaseException:
            deferred, self.deferred = (self.deferred, None)
            deferred.errback(failure.Failure())
        else:
            deferred, self.deferred = (self.deferred, None)
            deferred.callback(response)

    def clientConnectionLost(self, _, reason):
        if self.deferred is not None:
            deferred, self.deferred = (self.deferred, None)
            deferred.errback(reason)
    clientConnectionFailed = clientConnectionLost

    def badStatus(self, status, message):
        deferred, self.deferred = (self.deferred, None)
        deferred.errback(ValueError(status, message))