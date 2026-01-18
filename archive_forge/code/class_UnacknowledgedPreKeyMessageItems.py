from . import storageprotos_pb2 as storageprotos
from ..identitykeypair import IdentityKey, IdentityKeyPair
from ..ratchet.rootkey import RootKey
from ..kdf.hkdf import HKDF
from ..ecc.curve import Curve
from ..ecc.eckeypair import ECKeyPair
from ..ratchet.chainkey import ChainKey
from ..kdf.messagekeys import MessageKeys
class UnacknowledgedPreKeyMessageItems:

    def __init__(self, preKeyId, signedPreKeyId, baseKey):
        """
            :type preKeyId: int
            :type signedPreKeyId: int
            :type baseKey: ECPublicKey
            """
        self.preKeyId = preKeyId
        self.signedPreKeyId = signedPreKeyId
        self.baseKey = baseKey

    def getPreKeyId(self):
        return self.preKeyId

    def getSignedPreKeyId(self):
        return self.signedPreKeyId

    def getBaseKey(self):
        return self.baseKey