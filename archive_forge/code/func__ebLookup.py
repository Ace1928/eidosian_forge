import struct
from twisted.internet import defer
from twisted.protocols import basic
from twisted.python import failure, log
def _ebLookup(self, failure, sport, cport):
    if failure.check(IdentError):
        self.sendLine('%d, %d : ERROR : %s' % (sport, cport, failure.value))
    else:
        log.err(failure)
        self.sendLine('%d, %d : ERROR : %s' % (sport, cport, IdentError(failure.value)))