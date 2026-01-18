import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Cipher import ChaCha20_Poly1305
from Cryptodome.Hash import SHAKE128
from Cryptodome.Util.strxor import strxor
class TestVectorsRFC(unittest.TestCase):
    """Test cases from RFC7539"""
    test_vectors_hex = [('50 51 52 53 c0 c1 c2 c3 c4 c5 c6 c7', '4c 61 64 69 65 73 20 61 6e 64 20 47 65 6e 74 6c65 6d 65 6e 20 6f 66 20 74 68 65 20 63 6c 61 7373 20 6f 66 20 27 39 39 3a 20 49 66 20 49 20 636f 75 6c 64 20 6f 66 66 65 72 20 79 6f 75 20 6f6e 6c 79 20 6f 6e 65 20 74 69 70 20 66 6f 72 2074 68 65 20 66 75 74 75 72 65 2c 20 73 75 6e 7363 72 65 65 6e 20 77 6f 75 6c 64 20 62 65 20 6974 2e', 'd3 1a 8d 34 64 8e 60 db 7b 86 af bc 53 ef 7e c2a4 ad ed 51 29 6e 08 fe a9 e2 b5 a7 36 ee 62 d63d be a4 5e 8c a9 67 12 82 fa fb 69 da 92 72 8b1a 71 de 0a 9e 06 0b 29 05 d6 a5 b6 7e cd 3b 3692 dd bd 7f 2d 77 8b 8c 98 03 ae e3 28 09 1b 58fa b3 24 e4 fa d6 75 94 55 85 80 8b 48 31 d7 bc3f f4 de f0 8e 4b 7a 9d e5 76 d2 65 86 ce c6 4b61 16', '1a:e1:0b:59:4f:09:e2:6a:7e:90:2e:cb:d0:60:06:91', '80 81 82 83 84 85 86 87 88 89 8a 8b 8c 8d 8e 8f90 91 92 93 94 95 96 97 98 99 9a 9b 9c 9d 9e 9f', '07 00 00 00' + '40 41 42 43 44 45 46 47'), ('f3 33 88 86 00 00 00 00 00 00 4e 91', '49 6e 74 65 72 6e 65 74 2d 44 72 61 66 74 73 2061 72 65 20 64 72 61 66 74 20 64 6f 63 75 6d 656e 74 73 20 76 61 6c 69 64 20 66 6f 72 20 61 206d 61 78 69 6d 75 6d 20 6f 66 20 73 69 78 20 6d6f 6e 74 68 73 20 61 6e 64 20 6d 61 79 20 62 6520 75 70 64 61 74 65 64 2c 20 72 65 70 6c 61 6365 64 2c 20 6f 72 20 6f 62 73 6f 6c 65 74 65 6420 62 79 20 6f 74 68 65 72 20 64 6f 63 75 6d 656e 74 73 20 61 74 20 61 6e 79 20 74 69 6d 65 2e20 49 74 20 69 73 20 69 6e 61 70 70 72 6f 70 7269 61 74 65 20 74 6f 20 75 73 65 20 49 6e 74 6572 6e 65 74 2d 44 72 61 66 74 73 20 61 73 20 7265 66 65 72 65 6e 63 65 20 6d 61 74 65 72 69 616c 20 6f 72 20 74 6f 20 63 69 74 65 20 74 68 656d 20 6f 74 68 65 72 20 74 68 61 6e 20 61 73 202f e2 80 9c 77 6f 72 6b 20 69 6e 20 70 72 6f 6772 65 73 73 2e 2f e2 80 9d', '64 a0 86 15 75 86 1a f4 60 f0 62 c7 9b e6 43 bd5e 80 5c fd 34 5c f3 89 f1 08 67 0a c7 6c 8c b24c 6c fc 18 75 5d 43 ee a0 9e e9 4e 38 2d 26 b0bd b7 b7 3c 32 1b 01 00 d4 f0 3b 7f 35 58 94 cf33 2f 83 0e 71 0b 97 ce 98 c8 a8 4a bd 0b 94 8114 ad 17 6e 00 8d 33 bd 60 f9 82 b1 ff 37 c8 5597 97 a0 6e f4 f0 ef 61 c1 86 32 4e 2b 35 06 3836 06 90 7b 6a 7c 02 b0 f9 f6 15 7b 53 c8 67 e4b9 16 6c 76 7b 80 4d 46 a5 9b 52 16 cd e7 a4 e990 40 c5 a4 04 33 22 5e e2 82 a1 b0 a0 6c 52 3eaf 45 34 d7 f8 3f a1 15 5b 00 47 71 8c bc 54 6a0d 07 2b 04 b3 56 4e ea 1b 42 22 73 f5 48 27 1a0b b2 31 60 53 fa 76 99 19 55 eb d6 31 59 43 4ece bb 4e 46 6d ae 5a 10 73 a6 72 76 27 09 7a 1049 e6 17 d9 1d 36 10 94 fa 68 f0 ff 77 98 71 3030 5b ea ba 2e da 04 df 99 7b 71 4d 6c 6f 2c 29a6 ad 5c b4 02 2b 02 70 9b', 'ee ad 9d 67 89 0c bb 22 39 23 36 fe a1 85 1f 38', '1c 92 40 a5 eb 55 d3 8a f3 33 88 86 04 f6 b5 f047 39 17 c1 40 2b 80 09 9d ca 5c bc 20 70 75 c0', '00 00 00 00 01 02 03 04 05 06 07 08')]
    test_vectors = [[unhexlify(x.replace(' ', '').replace(':', '')) for x in tv] for tv in test_vectors_hex]

    def runTest(self):
        for assoc_data, pt, ct, mac, key, nonce in self.test_vectors:
            cipher = ChaCha20_Poly1305.new(key=key, nonce=nonce)
            cipher.update(assoc_data)
            ct2, mac2 = cipher.encrypt_and_digest(pt)
            self.assertEqual(ct, ct2)
            self.assertEqual(mac, mac2)
            cipher = ChaCha20_Poly1305.new(key=key, nonce=nonce)
            cipher.update(assoc_data)
            pt2 = cipher.decrypt_and_verify(ct, mac)
            self.assertEqual(pt, pt2)