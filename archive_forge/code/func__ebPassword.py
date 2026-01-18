import struct
from twisted.conch import error, interfaces
from twisted.conch.ssh import keys, service, transport
from twisted.conch.ssh.common import NS, getNS
from twisted.cred import credentials
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, reactor
from twisted.logger import Logger
from twisted.python import failure
from twisted.python.compat import nativeString
def _ebPassword(self, f):
    """
        If the password is invalid, wait before sending the failure in order
        to delay brute-force password guessing.
        """
    d = defer.Deferred()
    self.clock.callLater(self.passwordDelay, d.callback, f)
    return d