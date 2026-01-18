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
class _JellyableAvatarMixin:
    """
    Helper class for code which deals with avatars which PB must be capable of
    sending to a peer.
    """

    def _cbLogin(self, result):
        """
        Ensure that the avatar to be returned to the client is jellyable and
        set up disconnection notification to call the realm's logout object.
        """
        interface, avatar, logout = result
        if not IJellyable.providedBy(avatar):
            avatar = AsReferenceable(avatar, 'perspective')
        puid = avatar.processUniqueID()
        logout = [logout]

        def maybeLogout():
            if not logout:
                return
            fn = logout[0]
            del logout[0]
            fn()
        self.broker._localCleanup[puid] = maybeLogout
        self.broker.notifyOnDisconnect(maybeLogout)
        return avatar