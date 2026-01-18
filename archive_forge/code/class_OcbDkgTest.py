import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
class OcbDkgTest(unittest.TestCase):
    """Test vectors from https://gitlab.com/dkg/ocb-test-vectors"""

    def test_1_2(self):
        tvs = []
        for fi in (1, 2):
            for nb in (104, 112, 120):
                tv_file = load_test_vectors(('Cipher', 'AES'), 'test-vector-%d-nonce%d.txt' % (fi, nb), 'DKG tests, %d, %d bits' % (fi, nb), {})
                if tv_file is None:
                    break
                key = tv_file[0].k
                for tv in tv_file[1:]:
                    tv.k = key
                    tvs.append(tv)
        for tv in tvs:
            k, n, a, p, c = (tv.k, tv.n, tv.a, tv.p, tv.c)
            mac_len = len(c) - len(p)
            cipher = AES.new(k, AES.MODE_OCB, nonce=n, mac_len=mac_len)
            cipher.update(a)
            c_out, tag_out = cipher.encrypt_and_digest(p)
            self.assertEqual(c, c_out + tag_out)

    def test_3(self):

        def check(keylen, taglen, noncelen, exp):
            result = algo_rfc7253(keylen, taglen, noncelen)
            self.assertEqual(result, unhexlify(exp))
        check(128, 128, 104, 'C47F5F0341E15326D4D1C46F47F05062')
        check(192, 128, 104, '95B9167A38EB80495DFC561A8486E109')
        check(256, 128, 104, 'AFE1CDDB97028FD92F8FB3C8CFBA7D83')
        check(128, 96, 104, 'F471B4983BA80946DF217A54')
        check(192, 96, 104, '5AE828BC51C24D85FA5CC7B2')
        check(256, 96, 104, '8C8335982E2B734616CAD14C')
        check(128, 64, 104, 'B553F74B85FD1E5B')
        check(192, 64, 104, '3B49D20E513531F9')
        check(256, 64, 104, 'ED6DA5B1216BF8BB')
        check(128, 128, 112, 'CA8AFCA031BAC3F480A583BD6C50A547')
        check(192, 128, 112, 'D170C1DF356308079DA9A3F619147148')
        check(256, 128, 112, '57F94381F2F9231EFB04AECD323757C3')
        check(128, 96, 112, '3A618B2531ED39F260C750DC')
        check(192, 96, 112, '9071EB89FEDBADDA88FD286E')
        check(256, 96, 112, 'FDF0EFB97F21A39AC4BAB5AC')
        check(128, 64, 112, 'FAB2FF3A8DD82A13')
        check(192, 64, 112, 'AC01D912BD0737D3')
        check(256, 64, 112, '9D1FD0B500EA4ECF')
        check(128, 128, 120, '9E043A7140A25FB91F43BCC9DD7E0F46')
        check(192, 128, 120, '680000E53908323A7F396B955B8EC641')
        check(256, 128, 120, '8304B97FAACDA56E676602E1878A7E6F')
        check(128, 96, 120, '81F978AC9867E825D339847D')
        check(192, 96, 120, 'EFCF2D60B24926ADA48CF5B1')
        check(256, 96, 120, '84961DC56E917B165E58C174')
        check(128, 64, 120, '227AEE6C9D905A61')
        check(192, 64, 120, '541DE691B9E1A2F9')
        check(256, 64, 120, 'B0E761381C7129FC')

    def test_2_bugfix(self):
        nonce = unhexlify('EEDDCCBBAA9988776655443322110D')
        key = unhexlify('0F0E0D0C0B0A09080706050403020100')
        A = unhexlify('000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F2021222324252627')
        P = unhexlify('000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F2021222324252627')
        C = unhexlify('07E903BFC49552411ABC865F5ECE60F6FAD1F5A9F14D3070FA2F1308A563207FFE14C1EEA44B22059C7484319D8A2C53C236A7B3')
        mac_len = len(C) - len(P)
        buggy_result = unhexlify('BA015C4E5AE54D76C890AE81BD40DC5703EDC30E8AC2A58BC5D8FA4D61C5BAE6C39BEAC435B2FD56A2A5085C1B135D770C8264B7')
        cipher = AES.new(key, AES.MODE_OCB, nonce=nonce[:-1], mac_len=mac_len)
        cipher.update(A)
        C_out2, tag_out2 = cipher.encrypt_and_digest(P)
        self.assertEqual(buggy_result, C_out2 + tag_out2)