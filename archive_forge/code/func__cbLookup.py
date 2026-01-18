import struct
from twisted.internet import defer
from twisted.protocols import basic
from twisted.python import failure, log
def _cbLookup(self, result, sport, cport):
    sysName, userId = result
    self.sendLine('%d, %d : USERID : %s : %s' % (sport, cport, sysName, userId))