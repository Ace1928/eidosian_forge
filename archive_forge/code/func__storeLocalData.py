from axolotl.state.identitykeystore import IdentityKeyStore
from axolotl.identitykey import IdentityKey
from axolotl.identitykeypair import IdentityKeyPair
from axolotl.util.keyhelper import KeyHelper
from axolotl.ecc.djbec import *
import sys
def _storeLocalData(self, registrationId, identityKeyPair):
    q = 'INSERT INTO identities(recipient_id, registration_id, public_key, private_key) VALUES(-1, ?, ?, ?)'
    c = self.dbConn.cursor()
    pubKey = identityKeyPair.getPublicKey().getPublicKey().serialize()
    privKey = identityKeyPair.getPrivateKey().serialize()
    if sys.version_info < (2, 7):
        pubKey = buffer(pubKey)
        privKey = buffer(privKey)
    c.execute(q, (registrationId, pubKey, privKey))
    self.dbConn.commit()