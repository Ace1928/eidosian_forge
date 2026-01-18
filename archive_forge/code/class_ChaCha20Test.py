import os
import re
import unittest
from binascii import hexlify, unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.strxor import strxor_c
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import ChaCha20
class ChaCha20Test(unittest.TestCase):

    def test_new_positive(self):
        cipher = ChaCha20.new(key=b('0') * 32, nonce=b'0' * 8)
        self.assertEqual(cipher.nonce, b'0' * 8)
        cipher = ChaCha20.new(key=b('0') * 32, nonce=b'0' * 12)
        self.assertEqual(cipher.nonce, b'0' * 12)

    def test_new_negative(self):
        new = ChaCha20.new
        self.assertRaises(TypeError, new)
        self.assertRaises(TypeError, new, nonce=b('0'))
        self.assertRaises(ValueError, new, nonce=b('0') * 8, key=b('0'))
        self.assertRaises(ValueError, new, nonce=b('0'), key=b('0') * 32)

    def test_default_nonce(self):
        cipher1 = ChaCha20.new(key=bchr(1) * 32)
        cipher2 = ChaCha20.new(key=bchr(1) * 32)
        self.assertEqual(len(cipher1.nonce), 8)
        self.assertNotEqual(cipher1.nonce, cipher2.nonce)

    def test_nonce(self):
        key = b'A' * 32
        nonce1 = b'P' * 8
        cipher1 = ChaCha20.new(key=key, nonce=nonce1)
        self.assertEqual(nonce1, cipher1.nonce)
        nonce2 = b'Q' * 12
        cipher2 = ChaCha20.new(key=key, nonce=nonce2)
        self.assertEqual(nonce2, cipher2.nonce)

    def test_eiter_encrypt_or_decrypt(self):
        """Verify that a cipher cannot be used for both decrypting and encrypting"""
        c1 = ChaCha20.new(key=b('5') * 32, nonce=b('6') * 8)
        c1.encrypt(b('8'))
        self.assertRaises(TypeError, c1.decrypt, b('9'))
        c2 = ChaCha20.new(key=b('5') * 32, nonce=b('6') * 8)
        c2.decrypt(b('8'))
        self.assertRaises(TypeError, c2.encrypt, b('9'))

    def test_round_trip(self):
        pt = b('A') * 1024
        c1 = ChaCha20.new(key=b('5') * 32, nonce=b('6') * 8)
        c2 = ChaCha20.new(key=b('5') * 32, nonce=b('6') * 8)
        ct = c1.encrypt(pt)
        self.assertEqual(c2.decrypt(ct), pt)
        self.assertEqual(c1.encrypt(b('')), b(''))
        self.assertEqual(c2.decrypt(b('')), b(''))

    def test_streaming(self):
        """Verify that an arbitrary number of bytes can be encrypted/decrypted"""
        from Cryptodome.Hash import SHA1
        segments = (1, 3, 5, 7, 11, 17, 23)
        total = sum(segments)
        pt = b('')
        while len(pt) < total:
            pt += SHA1.new(pt).digest()
        cipher1 = ChaCha20.new(key=b('7') * 32, nonce=b('t') * 8)
        ct = cipher1.encrypt(pt)
        cipher2 = ChaCha20.new(key=b('7') * 32, nonce=b('t') * 8)
        cipher3 = ChaCha20.new(key=b('7') * 32, nonce=b('t') * 8)
        idx = 0
        for segment in segments:
            self.assertEqual(cipher2.decrypt(ct[idx:idx + segment]), pt[idx:idx + segment])
            self.assertEqual(cipher3.encrypt(pt[idx:idx + segment]), ct[idx:idx + segment])
            idx += segment

    def test_seek(self):
        cipher1 = ChaCha20.new(key=b('9') * 32, nonce=b('e') * 8)
        offset = 64 * 900 + 7
        pt = b('1') * 64
        cipher1.encrypt(b('0') * offset)
        ct1 = cipher1.encrypt(pt)
        cipher2 = ChaCha20.new(key=b('9') * 32, nonce=b('e') * 8)
        cipher2.seek(offset)
        ct2 = cipher2.encrypt(pt)
        self.assertEqual(ct1, ct2)

    def test_seek_tv(self):
        key = bchr(0) + bchr(255) + bchr(0) * 30
        nonce = bchr(0) * 8
        cipher = ChaCha20.new(key=key, nonce=nonce)
        cipher.seek(64 * 2)
        expected_key_stream = unhexlify(b('72d54dfbf12ec44b362692df94137f328fea8da73990265ec1bbbea1ae9af0ca13b25aa26cb4a648cb9b9d1be65b2c0924a66c54d545ec1b7374f4872e99f096'))
        ct = cipher.encrypt(bchr(0) * len(expected_key_stream))
        self.assertEqual(expected_key_stream, ct)

    def test_rfc7539(self):
        tvs = [('00' * 32, '00' * 12, 0, '00' * 16 * 4, '76b8e0ada0f13d90405d6ae55386bd28bdd219b8a08ded1aa836efcc8b770dc7da41597c5157488d7724e03fb8d84a376a43b8f41518a11cc387b669b2ee6586'), ('00' * 31 + '01', '00' * 11 + '02', 1, '416e79207375626d697373696f6e20746f20746865204945544620696e74656e6465642062792074686520436f6e7472696275746f7220666f72207075626c69636174696f6e20617320616c6c206f722070617274206f6620616e204945544620496e7465726e65742d4472616674206f722052464320616e6420616e792073746174656d656e74206d6164652077697468696e2074686520636f6e74657874206f6620616e204945544620616374697669747920697320636f6e7369646572656420616e20224945544620436f6e747269627574696f6e222e20537563682073746174656d656e747320696e636c756465206f72616c2073746174656d656e747320696e20494554462073657373696f6e732c2061732077656c6c206173207772697474656e20616e6420656c656374726f6e696320636f6d6d756e69636174696f6e73206d61646520617420616e792074696d65206f7220706c6163652c207768696368206172652061646472657373656420746f', 'a3fbf07df3fa2fde4f376ca23e82737041605d9f4f4f57bd8cff2c1d4b7955ec2a97948bd3722915c8f3d337f7d370050e9e96d647b7c39f56e031ca5eb6250d4042e02785ececfa4b4bb5e8ead0440e20b6e8db09d881a7c6132f420e52795042bdfa7773d8a9051447b3291ce1411c680465552aa6c405b7764d5e87bea85ad00f8449ed8f72d0d662ab052691ca66424bc86d2df80ea41f43abf937d3259dc4b2d0dfb48a6c9139ddd7f76966e928e635553ba76c5c879d7b35d49eb2e62b0871cdac638939e25e8a1e0ef9d5280fa8ca328b351c3c765989cbcf3daa8b6ccc3aaf9f3979c92b3720fc88dc95ed84a1be059c6499b9fda236e7e818b04b0bc39c1e876b193bfe5569753f88128cc08aaa9b63d1a16f80ef2554d7189c411f5869ca52c5b83fa36ff216b9c1d30062bebcfd2dc5bce0911934fda79a86f6e698ced759c3ff9b6477338f3da4f9cd8514ea9982ccafb341b2384dd902f3d1ab7ac61dd29c6f21ba5b862f3730e37cfdc4fd806c22f221'), ('1c9240a5eb55d38af333888604f6b5f0473917c1402b80099dca5cbc207075c0', '00' * 11 + '02', 42, '2754776173206272696c6c69672c20616e642074686520736c6974687920746f7665730a446964206779726520616e642067696d626c6520696e2074686520776162653a0a416c6c206d696d737920776572652074686520626f726f676f7665732c0a416e6420746865206d6f6d65207261746873206f757467726162652e', '62e6347f95ed87a45ffae7426f27a1df5fb69110044c0d73118effa95b01e5cf166d3df2d721caf9b21e5fb14c616871fd84c54f9d65b283196c7fe4f60553ebf39c6402c42234e32a356b3e764312a61a5532055716ead6962568f87d3f3f7704c6a8d1bcd1bf4d50d6154b6da731b187b58dfd728afa36757a797ac188d1')]
        for tv in tvs:
            key = unhexlify(tv[0])
            nonce = unhexlify(tv[1])
            offset = tv[2] * 64
            pt = unhexlify(tv[3])
            ct_expect = unhexlify(tv[4])
            cipher = ChaCha20.new(key=key, nonce=nonce)
            if offset != 0:
                cipher.seek(offset)
            ct = cipher.encrypt(pt)
            assert ct == ct_expect