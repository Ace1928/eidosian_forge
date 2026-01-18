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
def _cbGetPublicKey(self, publicKey):
    if not isinstance(publicKey, keys.Key):
        publicKey = None
    if publicKey is not None:
        self.lastPublicKey = publicKey
        self.triedPublicKeys.append(publicKey)
        self._log.debug('using key of type {keyType}', keyType=publicKey.type())
        self.askForAuth(b'publickey', b'\x00' + NS(publicKey.sshType()) + NS(publicKey.blob()))
        return True
    else:
        return False