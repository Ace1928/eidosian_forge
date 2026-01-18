import random
from hashlib import md5
from zope.interface import Interface, implementer
from twisted.cred.credentials import (
from twisted.cred.portal import Portal
from twisted.internet import defer, protocol
from twisted.persisted import styles
from twisted.python import failure, log, reflect
from twisted.python.compat import cmp, comparable
from twisted.python.components import registerAdapter
from twisted.spread import banana
from twisted.spread.flavors import (
from twisted.spread.interfaces import IJellyable, IUnjellyable
from twisted.spread.jelly import _newInstance, globalSecurity, jelly, unjelly
@implementer(IUsernameHashedPassword, IUsernameMD5Password)
class _PortalAuthChallenger(Referenceable, _JellyableAvatarMixin):
    """
    Called with response to password challenge.
    """

    def __init__(self, portal, broker, username, challenge):
        self.portal = portal
        self.broker = broker
        self.username = username
        self.challenge = challenge

    def remote_respond(self, response, mind):
        self.response = response
        d = self.portal.login(self, mind, IPerspective)
        d.addCallback(self._cbLogin)
        return d

    def checkPassword(self, password):
        """
        L{IUsernameHashedPassword}

        @param password: The password.
        @return: L{_PortalAuthChallenger.checkMD5Password}
        """
        return self.checkMD5Password(md5(password).digest())

    def checkMD5Password(self, md5Password):
        """
        L{IUsernameMD5Password}

        @param md5Password:
        @rtype: L{bool}
        @return: L{True} if password matches.
        """
        md = md5()
        md.update(md5Password)
        md.update(self.challenge)
        correct = md.digest()
        return self.response == correct