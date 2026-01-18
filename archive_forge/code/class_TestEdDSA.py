import unittest
from binascii import unhexlify
from Cryptodome.PublicKey import ECC
from Cryptodome.Signature import eddsa
from Cryptodome.Hash import SHA512, SHAKE256
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.number import bytes_to_long
class TestEdDSA(unittest.TestCase):

    def test_sign(self):
        for sk, _, msg, hashmod, ctx, exp_signature in rfc8032_tv_bytes:
            key = eddsa.import_private_key(sk)
            signer = eddsa.new(key, 'rfc8032', context=ctx)
            if hashmod is None:
                signature = signer.sign(msg)
            else:
                hashobj = hashmod.new(msg)
                signature = signer.sign(hashobj)
            self.assertEqual(exp_signature, signature)

    def test_verify(self):
        for _, pk, msg, hashmod, ctx, exp_signature in rfc8032_tv_bytes:
            key = eddsa.import_public_key(pk)
            verifier = eddsa.new(key, 'rfc8032', context=ctx)
            if hashmod is None:
                verifier.verify(msg, exp_signature)
            else:
                hashobj = hashmod.new(msg)
                verifier.verify(hashobj, exp_signature)

    def test_negative(self):
        key = ECC.generate(curve='ed25519')
        self.assertRaises(ValueError, eddsa.new, key, 'rfc9999')
        nist_key = ECC.generate(curve='p256')
        self.assertRaises(ValueError, eddsa.new, nist_key, 'rfc8032')