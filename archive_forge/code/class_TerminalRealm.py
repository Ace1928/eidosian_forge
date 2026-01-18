from typing import Dict
from zope.interface import implementer
from twisted.conch import avatar, error as econch, interfaces as iconch
from twisted.conch.insults import insults
from twisted.conch.ssh import factory, session
from twisted.python import components
class TerminalRealm:
    userFactory = TerminalUser
    sessionFactory = TerminalSession
    transportFactory = TerminalSessionTransport
    chainedProtocolFactory = insults.ServerProtocol

    def _getAvatar(self, avatarId):
        comp = components.Componentized()
        user = self.userFactory(comp, avatarId)
        sess = self.sessionFactory(comp)
        sess.transportFactory = self.transportFactory
        sess.chainedProtocolFactory = self.chainedProtocolFactory
        comp.setComponent(iconch.IConchUser, user)
        comp.setComponent(iconch.ISession, sess)
        return user

    def __init__(self, transportFactory=None):
        if transportFactory is not None:
            self.transportFactory = transportFactory

    def requestAvatar(self, avatarId, mind, *interfaces):
        for i in interfaces:
            if i is iconch.IConchUser:
                return (iconch.IConchUser, self._getAvatar(avatarId), lambda: None)
        raise NotImplementedError()