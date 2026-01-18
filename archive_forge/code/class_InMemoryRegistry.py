import socket
import time
import warnings
from collections import OrderedDict
from typing import Dict, List
from zope.interface import Interface, implementer
from twisted import cred
from twisted.internet import defer, protocol, reactor
from twisted.protocols import basic
from twisted.python import log
@implementer(IRegistry, ILocator)
class InMemoryRegistry:
    """
    A simplistic registry for a specific domain.
    """

    def __init__(self, domain):
        self.domain = domain
        self.users = {}

    def getAddress(self, userURI):
        if userURI.host != self.domain:
            return defer.fail(LookupError('unknown domain'))
        if userURI.username in self.users:
            dc, url = self.users[userURI.username]
            return defer.succeed(url)
        else:
            return defer.fail(LookupError('no such user'))

    def getRegistrationInfo(self, userURI):
        if userURI.host != self.domain:
            return defer.fail(LookupError('unknown domain'))
        if userURI.username in self.users:
            dc, url = self.users[userURI.username]
            return defer.succeed(Registration(int(dc.getTime() - time.time()), url))
        else:
            return defer.fail(LookupError('no such user'))

    def _expireRegistration(self, username):
        try:
            dc, url = self.users[username]
        except KeyError:
            return defer.fail(LookupError('no such user'))
        else:
            dc.cancel()
            del self.users[username]
        return defer.succeed(Registration(0, url))

    def registerAddress(self, domainURL, logicalURL, physicalURL):
        if domainURL.host != self.domain:
            log.msg("Registration for domain we don't handle.")
            return defer.fail(RegistrationError(404))
        if logicalURL.host != self.domain:
            log.msg("Registration for domain we don't handle.")
            return defer.fail(RegistrationError(404))
        if logicalURL.username in self.users:
            dc, old = self.users[logicalURL.username]
            dc.reset(3600)
        else:
            dc = reactor.callLater(3600, self._expireRegistration, logicalURL.username)
        log.msg(f'Registered {logicalURL.toString()} at {physicalURL.toString()}')
        self.users[logicalURL.username] = (dc, physicalURL)
        return defer.succeed(Registration(int(dc.getTime() - time.time()), physicalURL))

    def unregisterAddress(self, domainURL, logicalURL, physicalURL):
        return self._expireRegistration(logicalURL.username)