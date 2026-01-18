import unittest
import idna
class IDNATests(unittest.TestCase):

    def setUp(self):
        self.tld_strings = [['测试', b'xn--0zwm56d'], ['परीक्षा', b'xn--11b5bs3a9aj6g'], ['한국', b'xn--3e0b707e'], ['ভারত', b'xn--45brj9c'], ['বাংলা', b'xn--54b7fta0cc'], ['испытание', b'xn--80akhbyknj4f'], ['срб', b'xn--90a3ac'], ['테스트', b'xn--9t4b11yi5a'], ['சிங்கப்பூர்', b'xn--clchc0ea0b2g2a9gcd'], ['טעסט', b'xn--deba0ad'], ['中国', b'xn--fiqs8s'], ['中國', b'xn--fiqz9s'], ['భారత్', b'xn--fpcrj9c3d'], ['ලංකා', b'xn--fzc2c9e2c'], ['測試', b'xn--g6w251d'], ['ભારત', b'xn--gecrj9c'], ['भारत', b'xn--h2brj9c'], ['آزمایشی', b'xn--hgbk6aj7f53bba'], ['பரிட்சை', b'xn--hlcj6aya9esc7a'], ['укр', b'xn--j1amh'], ['香港', b'xn--j6w193g'], ['δοκιμή', b'xn--jxalpdlp'], ['إختبار', b'xn--kgbechtv'], ['台湾', b'xn--kprw13d'], ['台灣', b'xn--kpry57d'], ['الجزائر', b'xn--lgbbat1ad8j'], ['عمان', b'xn--mgb9awbf'], ['ایران', b'xn--mgba3a4f16a'], ['امارات', b'xn--mgbaam7a8h'], ['پاکستان', b'xn--mgbai9azgqp6j'], ['الاردن', b'xn--mgbayh7gpa'], ['بھارت', b'xn--mgbbh1a71e'], ['المغرب', b'xn--mgbc0a9azcg'], ['السعودية', b'xn--mgberp4a5d4ar'], ['გე', b'xn--node'], ['ไทย', b'xn--o3cw4h'], ['سورية', b'xn--ogbpf8fl'], ['рф', b'xn--p1ai'], ['تونس', b'xn--pgbs0dh'], ['ਭਾਰਤ', b'xn--s9brj9c'], ['مصر', b'xn--wgbh1c'], ['قطر', b'xn--wgbl6a'], ['இலங்கை', b'xn--xkc2al3hye2a'], ['இந்தியா', b'xn--xkc2dl3a5ee0h'], ['新加坡', b'xn--yfro4i67o'], ['فلسطين', b'xn--ygbi2ammx'], ['テスト', b'xn--zckzah'], ['қаз', b'xn--80ao21a'], ['مليسيا', b'xn--mgbx4cd0ab'], ['мон', b'xn--l1acc'], ['سودان', b'xn--mgbpl2fh']]

    def testIDNTLDALabels(self):
        for ulabel, alabel in self.tld_strings:
            self.assertEqual(alabel, idna.alabel(ulabel))

    def testIDNTLDULabels(self):
        for ulabel, alabel in self.tld_strings:
            self.assertEqual(ulabel, idna.ulabel(alabel))

    def test_valid_label_length(self):
        self.assertTrue(idna.valid_label_length('a' * 63))
        self.assertFalse(idna.valid_label_length('a' * 64))
        self.assertRaises(idna.IDNAError, idna.encode, 'a' * 64)

    def test_check_bidi(self):
        l = 'a'
        r = 'א'
        al = 'ا'
        an = '٠'
        en = '0'
        es = '-'
        cs = ','
        et = '$'
        on = '!'
        bn = '\u200c'
        nsm = 'ؐ'
        ws = ' '
        self.assertTrue(idna.check_bidi(l))
        self.assertTrue(idna.check_bidi(r))
        self.assertTrue(idna.check_bidi(al))
        self.assertRaises(idna.IDNABidiError, idna.check_bidi, an)
        self.assertTrue(idna.check_bidi(r + al))
        self.assertTrue(idna.check_bidi(r + al))
        self.assertTrue(idna.check_bidi(r + an))
        self.assertTrue(idna.check_bidi(r + en))
        self.assertTrue(idna.check_bidi(r + es + al))
        self.assertTrue(idna.check_bidi(r + cs + al))
        self.assertTrue(idna.check_bidi(r + et + al))
        self.assertTrue(idna.check_bidi(r + on + al))
        self.assertTrue(idna.check_bidi(r + bn + al))
        self.assertTrue(idna.check_bidi(r + nsm))
        self.assertRaises(idna.IDNABidiError, idna.check_bidi, r + l)
        self.assertRaises(idna.IDNABidiError, idna.check_bidi, r + ws)
        self.assertTrue(idna.check_bidi(r + al))
        self.assertTrue(idna.check_bidi(r + en))
        self.assertTrue(idna.check_bidi(r + an))
        self.assertTrue(idna.check_bidi(r + nsm))
        self.assertTrue(idna.check_bidi(r + nsm + nsm))
        self.assertRaises(idna.IDNABidiError, idna.check_bidi, r + on)
        self.assertTrue(idna.check_bidi(r + en))
        self.assertTrue(idna.check_bidi(r + an))
        self.assertRaises(idna.IDNABidiError, idna.check_bidi, r + en + an)
        self.assertRaises(idna.IDNABidiError, idna.check_bidi, r + an + en)
        self.assertTrue(idna.check_bidi(l + en, check_ltr=True))
        self.assertTrue(idna.check_bidi(l + es + l, check_ltr=True))
        self.assertTrue(idna.check_bidi(l + cs + l, check_ltr=True))
        self.assertTrue(idna.check_bidi(l + et + l, check_ltr=True))
        self.assertTrue(idna.check_bidi(l + on + l, check_ltr=True))
        self.assertTrue(idna.check_bidi(l + bn + l, check_ltr=True))
        self.assertTrue(idna.check_bidi(l + nsm, check_ltr=True))
        self.assertTrue(idna.check_bidi(l + l, check_ltr=True))
        self.assertTrue(idna.check_bidi(l + en, check_ltr=True))
        self.assertTrue(idna.check_bidi(l + en + nsm, check_ltr=True))
        self.assertTrue(idna.check_bidi(l + en + nsm + nsm, check_ltr=True))
        self.assertRaises(idna.IDNABidiError, idna.check_bidi, l + cs, check_ltr=True)

    def test_check_initial_combiner(self):
        m = '̀'
        a = 'a'
        self.assertTrue(idna.check_initial_combiner(a))
        self.assertTrue(idna.check_initial_combiner(a + m))
        self.assertRaises(idna.IDNAError, idna.check_initial_combiner, m + a)

    def test_check_hyphen_ok(self):
        self.assertTrue(idna.check_hyphen_ok('abc'))
        self.assertTrue(idna.check_hyphen_ok('a--b'))
        self.assertRaises(idna.IDNAError, idna.check_hyphen_ok, 'aa--')
        self.assertRaises(idna.IDNAError, idna.check_hyphen_ok, 'a-')
        self.assertRaises(idna.IDNAError, idna.check_hyphen_ok, '-a')

    def test_valid_contextj(self):
        zwnj = '\u200c'
        zwj = '\u200d'
        virama = '्'
        latin = 'a'
        self.assertFalse(idna.valid_contextj(zwnj, 0))
        self.assertFalse(idna.valid_contextj(latin + zwnj, 1))
        self.assertTrue(idna.valid_contextj(virama + zwnj, 1))
        self.assertFalse(idna.valid_contextj(zwj, 0))
        self.assertFalse(idna.valid_contextj(latin + zwj, 1))
        self.assertTrue(idna.valid_contextj(virama + zwj, 1))

    def test_valid_contexto(self):
        latin = 'a'
        latin_l = 'l'
        greek = 'α'
        hebrew = 'א'
        katakana = 'ァ'
        hiragana = 'ぁ'
        han = '漢'
        arabic_digit = '٠'
        ext_arabic_digit = '۰'
        latin_middle_dot = '·'
        self.assertTrue(idna.valid_contexto(latin_l + latin_middle_dot + latin_l, 1))
        self.assertFalse(idna.valid_contexto(latin_middle_dot + latin_l, 1))
        self.assertFalse(idna.valid_contexto(latin_l + latin_middle_dot, 0))
        self.assertFalse(idna.valid_contexto(latin_middle_dot, 0))
        self.assertFalse(idna.valid_contexto(latin_l + latin_middle_dot + latin, 1))
        glns = '͵'
        self.assertTrue(idna.valid_contexto(glns + greek, 0))
        self.assertFalse(idna.valid_contexto(glns + latin, 0))
        self.assertFalse(idna.valid_contexto(glns, 0))
        self.assertFalse(idna.valid_contexto(greek + glns, 1))
        geresh = '׳'
        self.assertTrue(idna.valid_contexto(hebrew + geresh, 1))
        self.assertFalse(idna.valid_contexto(latin + geresh, 1))
        gershayim = '״'
        self.assertTrue(idna.valid_contexto(hebrew + gershayim, 1))
        self.assertFalse(idna.valid_contexto(latin + gershayim, 1))
        ja_middle_dot = '・'
        self.assertTrue(idna.valid_contexto(katakana + ja_middle_dot + katakana, 1))
        self.assertTrue(idna.valid_contexto(hiragana + ja_middle_dot + hiragana, 1))
        self.assertTrue(idna.valid_contexto(han + ja_middle_dot + han, 1))
        self.assertTrue(idna.valid_contexto(han + ja_middle_dot + latin, 1))
        self.assertTrue(idna.valid_contexto('漢・字', 1))
        self.assertFalse(idna.valid_contexto('a・a', 1))
        self.assertTrue(idna.valid_contexto(arabic_digit + arabic_digit, 0))
        self.assertFalse(idna.valid_contexto(arabic_digit + ext_arabic_digit, 0))
        self.assertTrue(idna.valid_contexto(ext_arabic_digit + ext_arabic_digit, 0))
        self.assertFalse(idna.valid_contexto(ext_arabic_digit + arabic_digit, 0))

    def test_encode(self):
        self.assertEqual(idna.encode('xn--zckzah.xn--zckzah'), b'xn--zckzah.xn--zckzah')
        self.assertEqual(idna.encode('テスト.xn--zckzah'), b'xn--zckzah.xn--zckzah')
        self.assertEqual(idna.encode('テスト.テスト'), b'xn--zckzah.xn--zckzah')
        self.assertEqual(idna.encode('abc.abc'), b'abc.abc')
        self.assertEqual(idna.encode('xn--zckzah.abc'), b'xn--zckzah.abc')
        self.assertEqual(idna.encode('テスト.abc'), b'xn--zckzah.abc')
        self.assertEqual(idna.encode('ԡԥԣ-ԣԣ-----ԡԣԣԣ.aa'), b'xn---------90gglbagaar.aa')
        self.assertRaises(idna.IDNAError, idna.encode, 'ԡԤԣ-ԣԣ-----ԡԣԣԣ.aa', uts46=False)
        self.assertEqual(idna.encode('a' * 63), b'a' * 63)
        self.assertRaises(idna.IDNAError, idna.encode, 'a' * 64)
        self.assertRaises(idna.core.InvalidCodepoint, idna.encode, '*')

    def test_decode(self):
        self.assertEqual(idna.decode('xn--zckzah.xn--zckzah'), 'テスト.テスト')
        self.assertEqual(idna.decode('テスト.xn--zckzah'), 'テスト.テスト')
        self.assertEqual(idna.decode('テスト.テスト'), 'テスト.テスト')
        self.assertEqual(idna.decode('abc.abc'), 'abc.abc')
        self.assertEqual(idna.decode('xn---------90gglbagaar.aa'), 'ԡԥԣ-ԣԣ-----ԡԣԣԣ.aa')
        self.assertRaises(idna.IDNAError, idna.decode, 'XN---------90GGLBAGAAC.AA')
        self.assertRaises(idna.IDNAError, idna.decode, 'xn---------90gglbagaac.aa')
        self.assertRaises(idna.IDNAError, idna.decode, 'xn--')
        self.assertRaises(idna.IDNAError, idna.decode, b'\x8d\xd2')
        self.assertRaises(idna.IDNAError, idna.decode, b'A.A.0.a.a.A.0.a.A.A.0.a.A.0A.2.a.A.A.0.a.A.0.A.a.A0.a.a.A.0.a.fB.A.A.a.A.A.B.A.A.a.A.A.B.A.A.a.A.A.0.a.A.a.a.A.A.0.a.A.0.A.a.A0.a.a.A.0.a.fB.A.A.a.A.A.B.0A.A.a.A.A.B.A.A.a.A.A.a.A.A.B.A.A.a.A.0.a.B.A.A.a.A.B.A.a.A.A.5.a.A.0.a.Ba.A.B.A.A.a.A.0.a.Xn--B.A.A.A.a')