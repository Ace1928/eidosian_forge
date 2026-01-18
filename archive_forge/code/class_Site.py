import copy
import os
import re
import zlib
from binascii import hexlify
from html import escape
from typing import List, Optional
from urllib.parse import quote as _quote
from zope.interface import implementer
from incremental import Version
from twisted import copyright
from twisted.internet import address, interfaces
from twisted.internet.error import AlreadyCalled, AlreadyCancelled
from twisted.logger import Logger
from twisted.python import components, failure, reflect
from twisted.python.compat import nativeString, networkString
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.spread.pb import Copyable, ViewPoint
from twisted.web import http, iweb, resource, util
from twisted.web.error import UnsupportedMethod
from twisted.web.http import unquote
@implementer(interfaces.IProtocolNegotiationFactory)
class Site(http.HTTPFactory):
    """
    A web site: manage log, sessions, and resources.

    @ivar requestFactory: A factory which is called with (channel)
        and creates L{Request} instances. Default to L{Request}.

    @ivar displayTracebacks: If set, unhandled exceptions raised during
        rendering are returned to the client as HTML. Default to C{False}.

    @ivar sessionFactory: factory for sessions objects. Default to L{Session}.

    @ivar sessions: Mapping of session IDs to objects returned by
        C{sessionFactory}.
    @type sessions: L{dict} mapping L{bytes} to L{Session} given the default
        C{sessionFactory}

    @ivar counter: The number of sessions that have been generated.
    @type counter: L{int}

    @ivar sessionCheckTime: Deprecated and unused. See
        L{Session.sessionTimeout} instead.
    """
    counter = 0
    requestFactory = Request
    displayTracebacks = False
    sessionFactory = Session
    sessionCheckTime = 1800
    _entropy = os.urandom

    def __init__(self, resource, requestFactory=None, *args, **kwargs):
        """
        @param resource: The root of the resource hierarchy.  All request
            traversal for requests received by this factory will begin at this
            resource.
        @type resource: L{IResource} provider
        @param requestFactory: Overwrite for default requestFactory.
        @type requestFactory: C{callable} or C{class}.

        @see: L{twisted.web.http.HTTPFactory.__init__}
        """
        super().__init__(*args, **kwargs)
        self.sessions = {}
        self.resource = resource
        if requestFactory is not None:
            self.requestFactory = requestFactory

    def _openLogFile(self, path):
        from twisted.python import logfile
        return logfile.LogFile(os.path.basename(path), os.path.dirname(path))

    def __getstate__(self):
        d = self.__dict__.copy()
        d['sessions'] = {}
        return d

    def _mkuid(self):
        """
        (internal) Generate an opaque, unique ID for a user's session.
        """
        self.counter = self.counter + 1
        return hexlify(self._entropy(32))

    def makeSession(self):
        """
        Generate a new Session instance, and store it for future reference.
        """
        uid = self._mkuid()
        session = self.sessions[uid] = self.sessionFactory(self, uid)
        session.startCheckingExpiration()
        return session

    def getSession(self, uid):
        """
        Get a previously generated session.

        @param uid: Unique ID of the session.
        @type uid: L{bytes}.

        @raise KeyError: If the session is not found.
        """
        return self.sessions[uid]

    def buildProtocol(self, addr):
        """
        Generate a channel attached to this site.
        """
        channel = super().buildProtocol(addr)
        channel.requestFactory = self.requestFactory
        channel.site = self
        return channel
    isLeaf = 0

    def render(self, request):
        """
        Redirect because a Site is always a directory.
        """
        request.redirect(request.prePathURL() + b'/')
        request.finish()

    def getChildWithDefault(self, pathEl, request):
        """
        Emulate a resource's getChild method.
        """
        request.site = self
        return self.resource.getChildWithDefault(pathEl, request)

    def getResourceFor(self, request):
        """
        Get a resource for a request.

        This iterates through the resource hierarchy, calling
        getChildWithDefault on each resource it finds for a path element,
        stopping when it hits an element where isLeaf is true.
        """
        request.site = self
        request.sitepath = copy.copy(request.prepath)
        return resource.getChildForRequest(self.resource, request)

    def acceptableProtocols(self):
        """
        Protocols this server can speak.
        """
        baseProtocols = [b'http/1.1']
        if http.H2_ENABLED:
            baseProtocols.insert(0, b'h2')
        return baseProtocols