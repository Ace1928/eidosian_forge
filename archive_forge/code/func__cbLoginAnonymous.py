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
def _cbLoginAnonymous(self, root, client):
    """
        Attempt an anonymous login on the given remote root object.

        @type root: L{RemoteReference}
        @param root: The object on which to attempt the login, most likely
            returned by a call to L{PBClientFactory.getRootObject}.

        @param client: A jellyable object which will be used as the I{mind}
            parameter for the login attempt.

        @rtype: L{Deferred}
        @return: A L{Deferred} which will be called back with a
            L{RemoteReference} to an avatar when anonymous login succeeds, or
            which will errback if anonymous login fails.
        """
    return root.callRemote('loginAnonymous', client)