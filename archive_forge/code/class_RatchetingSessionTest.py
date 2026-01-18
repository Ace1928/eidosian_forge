import unittest
from ...identitykey import IdentityKey
from ...identitykeypair import IdentityKeyPair
from ...ecc.curve import Curve
from ...ecc.eckeypair import ECKeyPair
from ...ratchet.bobaxolotlparamaters import BobAxolotlParameters
from ...state.sessionstate import SessionState
from ...ratchet.ratchetingsession import RatchetingSession
class RatchetingSessionTest(unittest.TestCase):

    def test_ratchetingSessionAsBob(self):
        bobPublic = bytearray([5, 44, 180, 151, 118, 184, 119, 2, 5, 116, 90, 58, 110, 36, 245, 121, 205, 180, 186, 122, 137, 4, 16, 5, 146, 142, 187, 173, 201, 192, 90, 212, 88])
        bobPrivate = bytearray([161, 202, 180, 143, 124, 137, 63, 175, 169, 136, 10, 40, 195, 180, 153, 157, 40, 214, 50, 149, 98, 210, 122, 78, 164, 226, 46, 159, 241, 189, 214, 90])
        bobIdentityPublic = bytearray([5, 241, 244, 56, 116, 246, 150, 105, 86, 194, 221, 71, 63, 143, 161, 90, 222, 183, 29, 28, 185, 145, 178, 52, 22, 146, 50, 76, 239, 177, 197, 230, 38])
        bobIdentityPrivate = bytearray([72, 117, 204, 105, 221, 248, 234, 7, 25, 236, 148, 125, 97, 8, 17, 53, 134, 141, 95, 216, 1, 240, 44, 2, 37, 229, 22, 223, 33, 86, 96, 94])
        aliceBasePublic = bytearray([5, 71, 45, 31, 177, 169, 134, 44, 58, 246, 190, 172, 168, 146, 2, 119, 226, 178, 111, 74, 121, 33, 62, 199, 201, 6, 174, 179, 94, 3, 207, 137, 80])
        aliceEphemeralPublic = bytearray([5, 108, 62, 13, 31, 82, 2, 131, 239, 204, 85, 252, 165, 230, 112, 117, 185, 4, 0, 127, 24, 129, 209, 81, 175, 118, 223, 24, 197, 29, 41, 211, 75])
        aliceIdentityPublic = bytearray([5, 180, 168, 69, 86, 96, 173, 166, 91, 64, 16, 7, 246, 21, 230, 84, 4, 23, 70, 67, 46, 51, 57, 198, 135, 81, 73, 188, 238, 252, 180, 43, 74])
        bobSignedPreKeyPublic = bytearray([5, 172, 36, 138, 143, 38, 59, 230, 134, 53, 118, 235, 3, 98, 226, 140, 130, 143, 1, 7, 163, 55, 157, 52, 186, 177, 88, 107, 248, 199, 112, 205, 103])
        bobSignedPreKeyPrivate = bytearray([88, 57, 0, 19, 31, 183, 39, 153, 139, 120, 3, 254, 106, 194, 44, 197, 145, 243, 66, 228, 228, 42, 140, 141, 93, 120, 25, 66, 9, 184, 210, 83])
        senderChain = bytearray([151, 151, 202, 202, 83, 201, 137, 187, 226, 41, 164, 12, 167, 114, 112, 16, 235, 38, 4, 252, 20, 148, 93, 119, 149, 138, 10, 237, 160, 136, 180, 77])
        bobIdentityKeyPublic = IdentityKey(bobIdentityPublic, 0)
        bobIdentityKeyPrivate = Curve.decodePrivatePoint(bobIdentityPrivate)
        bobIdentityKey = IdentityKeyPair(bobIdentityKeyPublic, bobIdentityKeyPrivate)
        bobEphemeralPublicKey = Curve.decodePoint(bobPublic, 0)
        bobEphemeralPrivateKey = Curve.decodePrivatePoint(bobPrivate)
        bobEphemeralKey = ECKeyPair(bobEphemeralPublicKey, bobEphemeralPrivateKey)
        bobBaseKey = bobEphemeralKey
        bobSignedPreKey = ECKeyPair(Curve.decodePoint(bobSignedPreKeyPublic, 0), Curve.decodePrivatePoint(bobSignedPreKeyPrivate))
        aliceBasePublicKey = Curve.decodePoint(aliceBasePublic, 0)
        aliceEphemeralPublicKey = Curve.decodePoint(aliceEphemeralPublic, 0)
        aliceIdentityPublicKey = IdentityKey(aliceIdentityPublic, 0)
        parameters = BobAxolotlParameters.newBuilder().setOurIdentityKey(bobIdentityKey).setOurSignedPreKey(bobSignedPreKey).setOurRatchetKey(bobEphemeralKey).setOurOneTimePreKey(None).setTheirIdentityKey(aliceIdentityPublicKey).setTheirBaseKey(aliceBasePublicKey).create()
        session = SessionState()
        RatchetingSession.initializeSessionAsBob(session, parameters)
        self.assertEqual(session.getLocalIdentityKey(), bobIdentityKey.getPublicKey())
        self.assertEqual(session.getRemoteIdentityKey(), aliceIdentityPublicKey)
        self.assertEqual(session.getSenderChainKey().getKey(), senderChain)