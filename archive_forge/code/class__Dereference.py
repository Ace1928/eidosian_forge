from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from twisted.internet.defer import Deferred
class _Dereference(NotKnown):

    def __init__(self, id):
        NotKnown.__init__(self)
        self.id = id