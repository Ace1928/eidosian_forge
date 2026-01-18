import struct
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.ssh import keys
from twisted.conch.ssh.common import NS, getMP, getNS
from twisted.internet import defer, protocol
def _cbSignData(self, data):
    if ord(data[0:1]) != AGENT_SIGN_RESPONSE:
        raise ConchError('unexpected data: %i' % ord(data[0:1]))
    signature = getNS(data[1:])[0]
    return signature