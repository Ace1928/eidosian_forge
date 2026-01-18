from . import storageprotos_pb2 as storageprotos
from ..identitykeypair import IdentityKey, IdentityKeyPair
from ..ratchet.rootkey import RootKey
from ..kdf.hkdf import HKDF
from ..ecc.curve import Curve
from ..ecc.eckeypair import ECKeyPair
from ..ratchet.chainkey import ChainKey
from ..kdf.messagekeys import MessageKeys
def addReceiverChain(self, ECPublickKey_senderRatchetKey, chainKey):
    senderRatchetKey = ECPublickKey_senderRatchetKey
    chain = storageprotos.SessionStructure.Chain()
    chain.senderRatchetKey = senderRatchetKey.serialize()
    chain.chainKey.key = chainKey.getKey()
    chain.chainKey.index = chainKey.getIndex()
    self.sessionStructure.receiverChains.extend([chain])
    if len(self.sessionStructure.receiverChains) > 5:
        del self.sessionStructure.receiverChains[0]