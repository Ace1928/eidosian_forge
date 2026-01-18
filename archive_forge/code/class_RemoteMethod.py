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
@comparable
class RemoteMethod:
    """
    This is a translucent reference to a remote message.
    """

    def __init__(self, obj, name):
        """
        Initialize with a L{RemoteReference} and the name of this message.
        """
        self.obj = obj
        self.name = name

    def __cmp__(self, other):
        return cmp((self.obj, self.name), other)

    def __hash__(self):
        return hash((self.obj, self.name))

    def __call__(self, *args, **kw):
        """
        Asynchronously invoke a remote method.
        """
        return self.obj.broker._sendMessage(b'', self.obj.perspective, self.obj.luid, self.name.encode('utf-8'), args, kw)