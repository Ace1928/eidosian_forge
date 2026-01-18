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
def _cbSignedData(self, signedData):
    """
        Called back out of self.signData with the signed data.  Send the
        authentication request with the signature.

        @param signedData: the data signed by the user's private key.
        @type signedData: L{bytes}
        """
    publicKey = self.lastPublicKey
    self.askForAuth(b'publickey', b'\x01' + NS(publicKey.sshType()) + NS(publicKey.blob()) + NS(signedData))