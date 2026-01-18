from zope.interface import implementer
from twisted.internet import defer, error
from twisted.python import log
from twisted.python.failure import Failure
from twisted.spread import pb
from twisted.words.im import basesupport, interfaces
from twisted.words.im.locals import AWAY, OFFLINE, ONLINE
def _startLogOn(self, chatui):
    print('Connecting...', end=' ')
    d = pb.getObjectAt(self.host, self.port)
    d.addCallbacks(self._cbConnected, self._ebConnected, callbackArgs=(chatui,))
    return d