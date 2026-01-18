import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES, DES3
from Cryptodome.Hash import SHAKE128
from Cryptodome.Util.strxor import strxor
from Cryptodome.Cipher import DES, DES3, ARC2, CAST, Blowfish
class TestVectorsPaper(unittest.TestCase):
    """Class exercising the EAX test vectors found in
       http://www.cs.ucdavis.edu/~rogaway/papers/eax.pdf"""
    test_vectors_hex = [('6bfb914fd07eae6b', '', '', 'e037830e8389f27b025a2d6527e79d01', '233952dee4d5ed5f9b9c6d6ff80ff478', '62EC67F9C3A4A407FCB2A8C49031A8B3'), ('fa3bfd4806eb53fa', 'f7fb', '19dd', '5c4c9331049d0bdab0277408f67967e5', '91945d3f4dcbee0bf45ef52255f095a4', 'BECAF043B0A23D843194BA972C66DEBD'), ('234a3463c1264ac6', '1a47cb4933', 'd851d5bae0', '3a59f238a23e39199dc9266626c40f80', '01f74ad64077f2e704c0f60ada3dd523', '70C3DB4F0D26368400A10ED05D2BFF5E'), ('33cce2eabff5a79d', '481c9e39b1', '632a9d131a', 'd4c168a4225d8e1ff755939974a7bede', 'd07cf6cbb7f313bdde66b727afd3c5e8', '8408DFFF3C1A2B1292DC199E46B7D617'), ('aeb96eaebe2970e9', '40d0c07da5e4', '071dfe16c675', 'cb0677e536f73afe6a14b74ee49844dd', '35b6d0580005bbc12b0587124557d2c2', 'FDB6B06676EEDC5C61D74276E1F8E816'), ('d4482d1ca78dce0f', '4de3b35c3fc039245bd1fb7d', '835bb4f15d743e350e728414', 'abb8644fd6ccb86947c5e10590210a4f', 'bd8e6e11475e60b268784c38c62feb22', '6EAC5C93072D8E8513F750935E46DA1B'), ('65d2017990d62528', '8b0a79306c9ce7ed99dae4f87f8dd61636', '02083e3979da014812f59f11d52630da30', '137327d10649b0aa6e1c181db617d7f2', '7c77d6e813bed5ac98baa417477a2e7d', '1A8C98DCD73D38393B2BF1569DEEFC19'), ('54b9f04e6a09189a', '1bda122bce8a8dbaf1877d962b8592dd2d56', '2ec47b2c4954a489afc7ba4897edcdae8cc3', '3b60450599bd02c96382902aef7f832a', '5fff20cafab119ca2fc73549e20f5b0d', 'DDE59B97D722156D4D9AFF2BC7559826'), ('899a175897561d7e', '6cf36720872b8513f6eab1a8a44438d5ef11', '0de18fd0fdd91e7af19f1d8ee8733938b1e8', 'e7f6d2231618102fdb7fe55ff1991700', 'a4a4782bcffd3ec5e7ef6d8c34a56123', 'B781FCF2F75FA5A8DE97A9CA48E522EC'), ('126735fcc320d25a', 'ca40d7446e545ffaed3bd12a740a659ffbbb3ceab7', 'cb8920f87a6c75cff39627b56e3ed197c552d295a7', 'cfc46afc253b4652b1af3795b124ab6e', '8395fcf1e95bebd697bd010bc766aac3', '22E7ADD93CFC6393C57EC0B3C17D6B44')]
    test_vectors = [[unhexlify(x) for x in tv] for tv in test_vectors_hex]

    def runTest(self):
        for assoc_data, pt, ct, mac, key, nonce in self.test_vectors:
            cipher = AES.new(key, AES.MODE_EAX, nonce, mac_len=len(mac))
            cipher.update(assoc_data)
            ct2, mac2 = cipher.encrypt_and_digest(pt)
            self.assertEqual(ct, ct2)
            self.assertEqual(mac, mac2)
            cipher = AES.new(key, AES.MODE_EAX, nonce, mac_len=len(mac))
            cipher.update(assoc_data)
            pt2 = cipher.decrypt_and_verify(ct, mac)
            self.assertEqual(pt, pt2)