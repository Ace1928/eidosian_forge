import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TurboSHAKE128, TurboSHAKE256
from Cryptodome.Util.py3compat import bchr
class TurboSHAKE128TV(unittest.TestCase):

    def test_zero_1(self):
        tv = '1E 41 5F 1C 59 83 AF F2 16 92 17 27 7D 17 BB 53\n        8C D9 45 A3 97 DD EC 54 1F 1C E4 1A F2 C1 B7 4C'
        btv = txt2bin(tv)
        res = TurboSHAKE128.new().read(32)
        self.assertEqual(res, btv)

    def test_zero_2(self):
        tv = '1E 41 5F 1C 59 83 AF F2 16 92 17 27 7D 17 BB 53\n        8C D9 45 A3 97 DD EC 54 1F 1C E4 1A F2 C1 B7 4C\n        3E 8C CA E2 A4 DA E5 6C 84 A0 4C 23 85 C0 3C 15\n        E8 19 3B DF 58 73 73 63 32 16 91 C0 54 62 C8 DF'
        btv = txt2bin(tv)
        res = TurboSHAKE128.new().read(64)
        self.assertEqual(res, btv)

    def test_zero_3(self):
        tv = 'A3 B9 B0 38 59 00 CE 76 1F 22 AE D5 48 E7 54 DA\n        10 A5 24 2D 62 E8 C6 58 E3 F3 A9 23 A7 55 56 07'
        btv = txt2bin(tv)
        res = TurboSHAKE128.new().read(10032)[-32:]
        self.assertEqual(res, btv)

    def test_ptn_1(self):
        tv = '55 CE DD 6F 60 AF 7B B2 9A 40 42 AE 83 2E F3 F5\n        8D B7 29 9F 89 3E BB 92 47 24 7D 85 69 58 DA A9'
        btv = txt2bin(tv)
        res = TurboSHAKE128.new(data=ptn(1)).read(32)
        self.assertEqual(res, btv)

    def test_ptn_17(self):
        tv = '9C 97 D0 36 A3 BA C8 19 DB 70 ED E0 CA 55 4E C6\n        E4 C2 A1 A4 FF BF D9 EC 26 9C A6 A1 11 16 12 33'
        btv = txt2bin(tv)
        res = TurboSHAKE128.new(data=ptn(17)).read(32)
        self.assertEqual(res, btv)

    def test_ptn_17_2(self):
        tv = '96 C7 7C 27 9E 01 26 F7 FC 07 C9 B0 7F 5C DA E1\n        E0 BE 60 BD BE 10 62 00 40 E7 5D 72 23 A6 24 D2'
        btv = txt2bin(tv)
        res = TurboSHAKE128.new(data=ptn(17 ** 2)).read(32)
        self.assertEqual(res, btv)

    def test_ptn_17_3(self):
        tv = 'D4 97 6E B5 6B CF 11 85 20 58 2B 70 9F 73 E1 D6\n        85 3E 00 1F DA F8 0E 1B 13 E0 D0 59 9D 5F B3 72'
        btv = txt2bin(tv)
        res = TurboSHAKE128.new(data=ptn(17 ** 3)).read(32)
        self.assertEqual(res, btv)

    def test_ptn_17_4(self):
        tv = 'DA 67 C7 03 9E 98 BF 53 0C F7 A3 78 30 C6 66 4E\n        14 CB AB 7F 54 0F 58 40 3B 1B 82 95 13 18 EE 5C'
        btv = txt2bin(tv)
        data = ptn(17 ** 4)
        res = TurboSHAKE128.new(data=data).read(32)
        self.assertEqual(res, btv)
        xof = TurboSHAKE128.new()
        for x in data:
            xof.update(bchr(x))
        res = xof.read(32)
        self.assertEqual(res, btv)
        for chunk_size in (13, 17, 19, 23, 31):
            xof = TurboSHAKE128.new()
            for x in chunked(data, chunk_size):
                xof.update(x)
            res = xof.read(32)
            self.assertEqual(res, btv)

    def test_ptn_17_5(self):
        tv = 'B9 7A 90 6F BF 83 EF 7C 81 25 17 AB F3 B2 D0 AE\n        A0 C4 F6 03 18 CE 11 CF 10 39 25 12 7F 59 EE CD'
        btv = txt2bin(tv)
        data = ptn(17 ** 5)
        res = TurboSHAKE128.new(data=data).read(32)
        self.assertEqual(res, btv)
        xof = TurboSHAKE128.new()
        for chunk in chunked(data, 8192):
            xof.update(chunk)
        res = xof.read(32)
        self.assertEqual(res, btv)

    def test_ptn_17_6(self):
        tv = '35 CD 49 4A DE DE D2 F2 52 39 AF 09 A7 B8 EF 0C\n        4D 1C A4 FE 2D 1A C3 70 FA 63 21 6F E7 B4 C2 B1'
        btv = txt2bin(tv)
        data = ptn(17 ** 6)
        res = TurboSHAKE128.new(data=data).read(32)
        self.assertEqual(res, btv)

    def test_ffffff_d01(self):
        tv = 'BF 32 3F 94 04 94 E8 8E E1 C5 40 FE 66 0B E8 A0\n        C9 3F 43 D1 5E C0 06 99 84 62 FA 99 4E ED 5D AB'
        btv = txt2bin(tv)
        res = TurboSHAKE128.new(data=b'\xff\xff\xff', domain=1).read(32)
        self.assertEqual(res, btv)

    def test_ff_d06(self):
        tv = '8E C9 C6 64 65 ED 0D 4A 6C 35 D1 35 06 71 8D 68\n        7A 25 CB 05 C7 4C CA 1E 42 50 1A BD 83 87 4A 67'
        btv = txt2bin(tv)
        res = TurboSHAKE128.new(data=b'\xff', domain=6).read(32)
        self.assertEqual(res, btv)

    def test_ffffff_d07(self):
        tv = 'B6 58 57 60 01 CA D9 B1 E5 F3 99 A9 F7 77 23 BB\n        A0 54 58 04 2D 68 20 6F 72 52 68 2D BA 36 63 ED'
        btv = txt2bin(tv)
        res = TurboSHAKE128.new(data=b'\xff' * 3, domain=7).read(32)
        self.assertEqual(res, btv)

    def test_ffffffffffff_d0b(self):
        tv = '8D EE AA 1A EC 47 CC EE 56 9F 65 9C 21 DF A8 E1\n        12 DB 3C EE 37 B1 81 78 B2 AC D8 05 B7 99 CC 37'
        btv = txt2bin(tv)
        res = TurboSHAKE128.new(data=b'\xff' * 7, domain=11).read(32)
        self.assertEqual(res, btv)

    def test_ff_d30(self):
        tv = '55 31 22 E2 13 5E 36 3C 32 92 BE D2 C6 42 1F A2\n        32 BA B0 3D AA 07 C7 D6 63 66 03 28 65 06 32 5B'
        btv = txt2bin(tv)
        res = TurboSHAKE128.new(data=b'\xff', domain=48).read(32)
        self.assertEqual(res, btv)

    def test_ffffff_d7f(self):
        tv = '16 27 4C C6 56 D4 4C EF D4 22 39 5D 0F 90 53 BD\n        A6 D2 8E 12 2A BA 15 C7 65 E5 AD 0E 6E AF 26 F9'
        btv = txt2bin(tv)
        res = TurboSHAKE128.new(data=b'\xff' * 3, domain=127).read(32)
        self.assertEqual(res, btv)