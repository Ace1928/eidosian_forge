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
class RegisterProxy(Proxy):
    """
    A proxy that allows registration for a specific domain.

    Unregistered users won't be handled.
    """
    portal = None
    registry = None
    authorizers: Dict[str, IAuthorizer] = {}

    def __init__(self, *args, **kw):
        Proxy.__init__(self, *args, **kw)
        self.liveChallenges = {}

    def handle_ACK_request(self, message, host_port):
        host, port = host_port
        pass

    def handle_REGISTER_request(self, message, host_port):
        """
        Handle a registration request.

        Currently registration is not proxied.
        """
        host, port = host_port
        if self.portal is None:
            self.register(message, host, port)
        elif 'authorization' not in message.headers:
            return self.unauthorized(message, host, port)
        else:
            return self.login(message, host, port)

    def unauthorized(self, message, host, port):
        m = self.responseFromRequest(401, message)
        for scheme, auth in self.authorizers.items():
            chal = auth.getChallenge((host, port))
            if chal is None:
                value = f'{scheme.title()} realm="{self.host}"'
            else:
                value = f'{scheme.title()} {chal},realm="{self.host}"'
            m.headers.setdefault('www-authenticate', []).append(value)
        self.deliverResponse(m)

    def login(self, message, host, port):
        parts = message.headers['authorization'][0].split(None, 1)
        a = self.authorizers.get(parts[0].lower())
        if a:
            try:
                c = a.decode(parts[1])
            except SIPError:
                raise
            except BaseException:
                log.err()
                self.deliverResponse(self.responseFromRequest(500, message))
            else:
                c.username += '@' + self.host
                self.portal.login(c, None, IContact).addCallback(self._cbLogin, message, host, port).addErrback(self._ebLogin, message, host, port).addErrback(log.err)
        else:
            self.deliverResponse(self.responseFromRequest(501, message))

    def _cbLogin(self, i_a_l, message, host, port):
        i, a, l = i_a_l
        self.register(message, host, port)

    def _ebLogin(self, failure, message, host, port):
        failure.trap(cred.error.UnauthorizedLogin)
        self.unauthorized(message, host, port)

    def register(self, message, host, port):
        """
        Allow all users to register
        """
        name, toURL, params = parseAddress(message.headers['to'][0], clean=1)
        contact = None
        if 'contact' in message.headers:
            contact = message.headers['contact'][0]
        if message.headers.get('expires', [None])[0] == '0':
            self.unregister(message, toURL, contact)
        else:
            if contact is not None:
                name, contactURL, params = parseAddress(contact, host=host, port=port)
                d = self.registry.registerAddress(message.uri, toURL, contactURL)
            else:
                d = self.registry.getRegistrationInfo(toURL)
            d.addCallbacks(self._cbRegister, self._ebRegister, callbackArgs=(message,), errbackArgs=(message,))

    def _cbRegister(self, registration, message):
        response = self.responseFromRequest(200, message)
        if registration.contactURL != None:
            response.addHeader('contact', registration.contactURL.toString())
            response.addHeader('expires', '%d' % registration.secondsToExpiry)
        response.addHeader('content-length', '0')
        self.deliverResponse(response)

    def _ebRegister(self, error, message):
        error.trap(RegistrationError, LookupError)

    def unregister(self, message, toURL, contact):
        try:
            expires = int(message.headers['expires'][0])
        except ValueError:
            self.deliverResponse(self.responseFromRequest(400, message))
        else:
            if expires == 0:
                if contact == '*':
                    contactURL = '*'
                else:
                    name, contactURL, params = parseAddress(contact)
                d = self.registry.unregisterAddress(message.uri, toURL, contactURL)
                d.addCallback(self._cbUnregister, message).addErrback(self._ebUnregister, message)

    def _cbUnregister(self, registration, message):
        msg = self.responseFromRequest(200, message)
        msg.headers.setdefault('contact', []).append(registration.contactURL.toString())
        msg.addHeader('expires', '0')
        self.deliverResponse(msg)

    def _ebUnregister(self, registration, message):
        pass