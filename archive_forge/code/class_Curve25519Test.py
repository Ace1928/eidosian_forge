import unittest
from ..ecc.curve import Curve
from ..util.keyhelper import KeyHelper
class Curve25519Test(unittest.TestCase):

    def test_agreement(self):
        alicePublic = bytearray([5, 27, 183, 89, 102, 242, 233, 58, 54, 145, 223, 255, 148, 43, 178, 164, 102, 161, 192, 139, 141, 120, 202, 63, 77, 109, 248, 184, 191, 162, 228, 238, 40])
        alicePrivate = bytearray([200, 6, 67, 157, 201, 210, 196, 118, 255, 237, 143, 37, 128, 192, 136, 141, 88, 171, 64, 107, 247, 174, 54, 152, 135, 144, 33, 185, 107, 180, 191, 89])
        bobPublic = bytearray([5, 101, 54, 20, 153, 61, 43, 21, 238, 158, 95, 211, 216, 108, 231, 25, 239, 78, 193, 218, 174, 24, 134, 168, 123, 63, 95, 169, 86, 90, 39, 162, 47])
        bobPrivate = bytearray([176, 59, 52, 195, 58, 28, 68, 242, 37, 182, 98, 210, 191, 72, 89, 184, 19, 84, 17, 250, 123, 3, 134, 212, 95, 183, 93, 197, 185, 27, 68, 102])
        shared = bytearray([50, 95, 35, 147, 40, 148, 28, 237, 110, 103, 59, 134, 186, 65, 1, 116, 72, 233, 155, 100, 154, 156, 56, 6, 193, 221, 124, 164, 196, 119, 230, 41])
        alicePublicKey = Curve.decodePoint(alicePublic, 0)
        alicePrivateKey = Curve.decodePrivatePoint(alicePrivate)
        bobPublicKey = Curve.decodePoint(bobPublic, 0)
        bobPrivateKey = Curve.decodePrivatePoint(bobPrivate)
        sharedOne = Curve.calculateAgreement(alicePublicKey, bobPrivateKey)
        sharedTwo = Curve.calculateAgreement(bobPublicKey, alicePrivateKey)
        self.assertEqual(sharedOne, shared)
        self.assertEqual(sharedTwo, shared)

    def test_randomAgreements(self):
        for i in range(0, 50):
            alice = Curve.generateKeyPair()
            bob = Curve.generateKeyPair()
            sharedAlice = Curve.calculateAgreement(bob.getPublicKey(), alice.getPrivateKey())
            sharedBob = Curve.calculateAgreement(alice.getPublicKey(), bob.getPrivateKey())
            self.assertEqual(sharedAlice, sharedBob)

    def test_gensig(self):
        identityKeyPair = KeyHelper.generateIdentityKeyPair()
        KeyHelper.generateSignedPreKey(identityKeyPair, 0)

    def test_signature(self):
        aliceIdentityPublic = bytearray([5, 171, 126, 113, 125, 74, 22, 59, 125, 154, 29, 128, 113, 223, 233, 220, 248, 205, 205, 28, 234, 51, 57, 182, 53, 107, 232, 77, 136, 126, 50, 44, 100])
        aliceEphemeralPublic = bytearray([5, 237, 206, 157, 156, 65, 92, 167, 140, 183, 37, 46, 114, 194, 196, 165, 84, 211, 235, 41, 72, 90, 14, 29, 80, 49, 24, 209, 168, 45, 153, 251, 74])
        aliceSignature = bytearray([93, 232, 140, 169, 168, 155, 74, 17, 93, 167, 145, 9, 198, 124, 156, 116, 100, 163, 228, 24, 2, 116, 241, 203, 140, 99, 194, 152, 78, 40, 109, 251, 237, 232, 45, 235, 157, 205, 159, 174, 11, 251, 184, 33, 86, 155, 61, 144, 1, 189, 129, 48, 205, 17, 212, 134, 206, 240, 71, 189, 96, 184, 110, 136])
        alicePublicKey = Curve.decodePoint(aliceIdentityPublic, 0)
        aliceEphemeral = Curve.decodePoint(aliceEphemeralPublic, 0)
        res = Curve.verifySignature(alicePublicKey, aliceEphemeral.serialize(), bytes(aliceSignature))
        self.assertTrue(res)