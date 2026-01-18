import base64
import binascii
import os
import random
import re
import socket
import time
import warnings
from email.utils import parseaddr
from io import BytesIO
from typing import Type
from zope.interface import implementer
from twisted import cred
from twisted.copyright import longversion
from twisted.internet import defer, error, protocol, reactor
from twisted.internet._idna import _idnaText
from twisted.internet.interfaces import ISSLTransport, ITLSTransport
from twisted.mail._cred import (
from twisted.mail._except import (
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log, util
from twisted.python.compat import iterbytes, nativeString, networkString
from twisted.python.runtime import platform
import codecs
class ESMTP(SMTP):
    ctx = None
    canStartTLS = False
    startedTLS = False
    authenticated = False

    def __init__(self, chal=None, contextFactory=None):
        SMTP.__init__(self)
        if chal is None:
            chal = {}
        self.challengers = chal
        self.authenticated = False
        self.ctx = contextFactory

    def connectionMade(self):
        SMTP.connectionMade(self)
        self.canStartTLS = ITLSTransport.providedBy(self.transport)
        self.canStartTLS = self.canStartTLS and self.ctx is not None

    def greeting(self):
        return SMTP.greeting(self) + b' ESMTP'

    def extensions(self):
        """
        SMTP service extensions

        @return: the SMTP service extensions that are supported.
        @rtype: L{dict} with L{bytes} keys and a value of either L{None} or a
            L{list} of L{bytes}.
        """
        ext = {b'AUTH': list(self.challengers.keys())}
        if self.canStartTLS and (not self.startedTLS):
            ext[b'STARTTLS'] = None
        return ext

    def lookupMethod(self, command):
        command = nativeString(command)
        m = SMTP.lookupMethod(self, command)
        if m is None:
            m = getattr(self, 'ext_' + command.upper(), None)
        return m

    def listExtensions(self):
        r = []
        for c, v in self.extensions().items():
            if v is not None:
                if v:
                    r.append(c + b' ' + b' '.join(v))
            else:
                r.append(c)
        return b'\n'.join(r)

    def do_EHLO(self, rest):
        peer = self.transport.getPeer().host
        if not isinstance(peer, bytes):
            peer = peer.encode('idna')
        self._helo = (rest, peer)
        self._from = None
        self._to = []
        self.sendCode(250, self.host + b' Hello ' + peer + b', nice to meet you\n' + self.listExtensions())

    def ext_STARTTLS(self, rest):
        if self.startedTLS:
            self.sendCode(503, b'TLS already negotiated')
        elif self.ctx and self.canStartTLS:
            self.sendCode(220, b'Begin TLS negotiation now')
            self.transport.startTLS(self.ctx)
            self.startedTLS = True
        else:
            self.sendCode(454, b'TLS not available')

    def ext_AUTH(self, rest):
        if self.authenticated:
            self.sendCode(503, b'Already authenticated')
            return
        parts = rest.split(None, 1)
        chal = self.challengers.get(parts[0].upper(), lambda: None)()
        if not chal:
            self.sendCode(504, b'Unrecognized authentication type')
            return
        self.mode = AUTH
        self.challenger = chal
        if len(parts) > 1:
            chal.getChallenge()
            rest = parts[1]
        else:
            rest = None
        self.state_AUTH(rest)

    def _cbAuthenticated(self, loginInfo):
        """
        Save the state resulting from a successful cred login and mark this
        connection as authenticated.
        """
        result = SMTP._cbAnonymousAuthentication(self, loginInfo)
        self.authenticated = True
        return result

    def _ebAuthenticated(self, reason):
        """
        Handle cred login errors by translating them to the SMTP authenticate
        failed.  Translate all other errors into a generic SMTP error code and
        log the failure for inspection.  Stop all errors from propagating.

        @param reason: Reason for failure.
        """
        self.challenge = None
        if reason.check(cred.error.UnauthorizedLogin):
            self.sendCode(535, b'Authentication failed')
        else:
            log.err(reason, 'SMTP authentication failure')
            self.sendCode(451, b'Requested action aborted: local error in processing')

    def state_AUTH(self, response):
        """
        Handle one step of challenge/response authentication.

        @param response: The text of a response. If None, this
        function has been called as a result of an AUTH command with
        no initial response. A response of '*' aborts authentication,
        as per RFC 2554.
        """
        if self.portal is None:
            self.sendCode(454, b'Temporary authentication failure')
            self.mode = COMMAND
            return
        if response is None:
            challenge = self.challenger.getChallenge()
            encoded = base64.b64encode(challenge)
            self.sendCode(334, encoded)
            return
        if response == b'*':
            self.sendCode(501, b'Authentication aborted')
            self.challenger = None
            self.mode = COMMAND
            return
        try:
            uncoded = base64.b64decode(response)
        except (TypeError, binascii.Error):
            self.sendCode(501, b'Syntax error in parameters or arguments')
            self.challenger = None
            self.mode = COMMAND
            return
        self.challenger.setResponse(uncoded)
        if self.challenger.moreChallenges():
            challenge = self.challenger.getChallenge()
            coded = base64.b64encode(challenge)
            self.sendCode(334, coded)
            return
        self.mode = COMMAND
        result = self.portal.login(self.challenger, None, IMessageDeliveryFactory, IMessageDelivery)
        result.addCallback(self._cbAuthenticated)
        result.addCallback(lambda ign: self.sendCode(235, b'Authentication successful.'))
        result.addErrback(self._ebAuthenticated)