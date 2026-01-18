from zope.interface import implementer
from twisted.internet import defer, interfaces
from twisted.protocols import basic
from twisted.python.failure import Failure
from twisted.spread import pb
class LocalMethod:

    def __init__(self, local, name):
        self.local = local
        self.name = name

    def __call__(self, *args, **kw):
        return self.local.callRemote(self.name, *args, **kw)