import re
import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, bchr
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Hash import SHA1, HMAC, SHA256, MD5, SHA224, SHA384, SHA512
from Cryptodome.Cipher import AES, DES3
from Cryptodome.Protocol.KDF import (PBKDF1, PBKDF2, _S2V, HKDF, scrypt,
from Cryptodome.Protocol.KDF import _bcrypt_decode
class bcrypt_Tests(unittest.TestCase):

    def test_negative_cases(self):
        self.assertRaises(ValueError, bcrypt, b'1' * 73, 10)
        self.assertRaises(ValueError, bcrypt, b'1' * 10, 3)
        self.assertRaises(ValueError, bcrypt, b'1' * 10, 32)
        self.assertRaises(ValueError, bcrypt, b'1' * 10, 4, salt=b'')
        self.assertRaises(ValueError, bcrypt, b'1' * 10, 4, salt=b'1')
        self.assertRaises(ValueError, bcrypt, b'1' * 10, 4, salt=b'1' * 17)
        self.assertRaises(ValueError, bcrypt, b'1\x00' * 10, 4)

    def test_bytearray_mismatch(self):
        ref = bcrypt('pwd', 4)
        bcrypt_check('pwd', ref)
        bref = bytearray(ref)
        bcrypt_check('pwd', bref)
        wrong = ref[:-1] + bchr(bref[-1] ^ 1)
        self.assertRaises(ValueError, bcrypt_check, 'pwd', wrong)
        wrong = b'x' + ref[1:]
        self.assertRaises(ValueError, bcrypt_check, 'pwd', wrong)

    def test_empty_password(self):
        tvs = [(b'', 4, b'zVHmKQtGGQob.b/Nc7l9NO', b'$2a$04$zVHmKQtGGQob.b/Nc7l9NO8UlrYcW05FiuCj/SxsFO/ZtiN9.mNzy'), (b'', 5, b'zVHmKQtGGQob.b/Nc7l9NO', b'$2a$05$zVHmKQtGGQob.b/Nc7l9NOWES.1hkVBgy5IWImh9DOjKNU8atY4Iy'), (b'', 6, b'zVHmKQtGGQob.b/Nc7l9NO', b'$2a$06$zVHmKQtGGQob.b/Nc7l9NOjOl7l4oz3WSh5fJ6414Uw8IXRAUoiaO'), (b'', 7, b'zVHmKQtGGQob.b/Nc7l9NO', b'$2a$07$zVHmKQtGGQob.b/Nc7l9NOBsj1dQpBA1HYNGpIETIByoNX9jc.hOi'), (b'', 8, b'zVHmKQtGGQob.b/Nc7l9NO', b'$2a$08$zVHmKQtGGQob.b/Nc7l9NOiLTUh/9MDpX86/DLyEzyiFjqjBFePgO')]
        for idx, (password, cost, salt64, result) in enumerate(tvs):
            x = bcrypt(password, cost, salt=_bcrypt_decode(salt64))
            self.assertEqual(x, result)
            bcrypt_check(password, result)

    def test_random_password_and_salt_short_pw(self):
        tvs = [(b"<.S.2K(Zq'", 4, b'VYAclAMpaXY/oqAo9yUpku', b'$2a$04$VYAclAMpaXY/oqAo9yUpkuWmoYywaPzyhu56HxXpVltnBIfmO9tgu'), (b'5.rApO%5jA', 5, b'kVNDrnYKvbNr5AIcxNzeIu', b'$2a$05$kVNDrnYKvbNr5AIcxNzeIuRcyIF5cZk6UrwHGxENbxP5dVv.WQM/G'), (b'oW++kSrQW^', 6, b'QLKkRMH9Am6irtPeSKN5sO', b'$2a$06$QLKkRMH9Am6irtPeSKN5sObJGr3j47cO6Pdf5JZ0AsJXuze0IbsNm'), (b'ggJ\\KbTnDG', 7, b'4H896R09bzjhapgCPS/LYu', b'$2a$07$4H896R09bzjhapgCPS/LYuMzAQluVgR5iu/ALF8L8Aln6lzzYXwbq'), (b'49b0:;VkH/', 8, b'hfvO2retKrSrx5f2RXikWe', b'$2a$08$hfvO2retKrSrx5f2RXikWeFWdtSesPlbj08t/uXxCeZoHRWDz/xFe'), (b">9N^5jc##'", 9, b'XZLvl7rMB3EvM0c1.JHivu', b'$2a$09$XZLvl7rMB3EvM0c1.JHivuIDPJWeNJPTVrpjZIEVRYYB/mF6cYgJK'), (b'\\$ch)s4WXp', 10, b'aIjpMOLK5qiS9zjhcHR5TO', b'$2a$10$aIjpMOLK5qiS9zjhcHR5TOU7v2NFDmcsBmSFDt5EHOgp/jeTF3O/q'), (b'RYoj\\_>2P7', 12, b'esIAHiQAJNNBrsr5V13l7.', b'$2a$12$esIAHiQAJNNBrsr5V13l7.RFWWJI2BZFtQlkFyiWXjou05GyuREZa')]
        for idx, (password, cost, salt64, result) in enumerate(tvs):
            x = bcrypt(password, cost, salt=_bcrypt_decode(salt64))
            self.assertEqual(x, result)
            bcrypt_check(password, result)

    def test_random_password_and_salt_long_pw(self):
        tvs = [(b'^Q&"]A`%/A(BVGt>QaX0M-#<Q148&f', 4, b'vrRP5vQxyD4LrqiLd/oWRO', b'$2a$04$vrRP5vQxyD4LrqiLd/oWROgrrGINsw3gb4Ga5x2sn01jNmiLVECl6'), (b'nZa!rRf\\U;OL;R?>1ghq_+":Y0CRmY', 5, b'YuQvhokOGVnevctykUYpKu', b'$2a$05$YuQvhokOGVnevctykUYpKutZD2pWeGGYn3auyLOasguMY3/0BbIyq'), (b"F%uN/j>[GuB7-jB'_Yj!Tnb7Y!u^6)", 6, b'5L3vpQ0tG9O7k5gQ8nAHAe', b'$2a$06$5L3vpQ0tG9O7k5gQ8nAHAe9xxQiOcOLh8LGcI0PLWhIznsDt.S.C6'), (b'Z>BobP32ub"Cfe*Q<<WUq3rc=[GJr-', 7, b'hp8IdLueqE6qFh1zYycUZ.', b'$2a$07$hp8IdLueqE6qFh1zYycUZ.twmUH8eSTPQAEpdNXKMlwms9XfKqfea'), (b"Ik&8N['7*[1aCc1lOm8\\jWeD*H$eZM", 8, b'2ANDTYCB9m7vf0Prh7rSru', b'$2a$08$2ANDTYCB9m7vf0Prh7rSrupqpO3jJOkIz2oW/QHB4lCmK7qMytGV6'), (b'O)=%3[E$*q+>-q-=tRSjOBh8\\mLNW.', 9, b'nArqOfdCsD9kIbVnAixnwe', b'$2a$09$nArqOfdCsD9kIbVnAixnwe6s8QvyPYWtQBpEXKir2OJF9/oNBsEFe'), (b'/MH51`!BP&0tj3%YCA;Xk%e3S`o\\EI', 10, b'ePiAc.s.yoBi3B6p1iQUCe', b'$2a$10$ePiAc.s.yoBi3B6p1iQUCezn3mraLwpVJ5XGelVyYFKyp5FZn/y.u'), (b'ptAP"mcg6oH.";c0U2_oll.OKi<!ku', 12, b'aroG/pwwPj1tU5fl9a9pkO', b'$2a$12$aroG/pwwPj1tU5fl9a9pkO4rydAmkXRj/LqfHZOSnR6LGAZ.z.jwa')]
        for idx, (password, cost, salt64, result) in enumerate(tvs):
            x = bcrypt(password, cost, salt=_bcrypt_decode(salt64))
            self.assertEqual(x, result)
            bcrypt_check(password, result)

    def test_same_password_and_random_salt(self):
        tvs = [(b'Q/A:k3DP;X@=<0"hg&9c', 4, b'wbgDTvLMtyjQlNK7fjqwyO', b'$2a$04$wbgDTvLMtyjQlNK7fjqwyOakBoACQuYh11.VsKNarF4xUIOBWgD6S'), (b'Q/A:k3DP;X@=<0"hg&9c', 5, b'zbAaOmloOhxiKItjznRqru', b'$2a$05$zbAaOmloOhxiKItjznRqrunRqHlu3MAa7pMGv26Rr3WwyfGcwoRm6'), (b'Q/A:k3DP;X@=<0"hg&9c', 6, b'aOK0bWUvLI0qLkc3ti5jyu', b'$2a$06$aOK0bWUvLI0qLkc3ti5jyuAIQoqRzuqoK09kQqQ6Ou/YKDhW50/qa')]
        for idx, (password, cost, salt64, result) in enumerate(tvs):
            x = bcrypt(password, cost, salt=_bcrypt_decode(salt64))
            self.assertEqual(x, result)
            bcrypt_check(password, result)

    def test_same_password_and_salt_increasing_cost_factor(self):
        tvs = [(b"o<&+X'F4AQ8H,LU,N`&r", 4, b'BK5u.QHk1Driey7bvnFTH.', b'$2a$04$BK5u.QHk1Driey7bvnFTH.3smGwxd91PtoK2GxH5nZ7pcBsYX4lMq'), (b"o<&+X'F4AQ8H,LU,N`&r", 5, b'BK5u.QHk1Driey7bvnFTH.', b'$2a$05$BK5u.QHk1Driey7bvnFTH.t5P.jZvFBMzDB1IY4PwkkRPOyVbEtFG'), (b"o<&+X'F4AQ8H,LU,N`&r", 6, b'BK5u.QHk1Driey7bvnFTH.', b'$2a$06$BK5u.QHk1Driey7bvnFTH.6Ea1Z5db2p25CPXZbxb/3OyKQagg3pa'), (b"o<&+X'F4AQ8H,LU,N`&r", 7, b'BK5u.QHk1Driey7bvnFTH.', b'$2a$07$BK5u.QHk1Driey7bvnFTH.sruuQi8Lhv/0LWKDvNp3AGFk7ltdkm6'), (b"o<&+X'F4AQ8H,LU,N`&r", 8, b'BK5u.QHk1Driey7bvnFTH.', b'$2a$08$BK5u.QHk1Driey7bvnFTH.IE7KsaUzc4m7gzAMlyUPUeiYyACWe0q'), (b"o<&+X'F4AQ8H,LU,N`&r", 9, b'BK5u.QHk1Driey7bvnFTH.', b'$2a$09$BK5u.QHk1Driey7bvnFTH.1v4Xj1dwkp44QNg0cVAoQt4FQMMrvnS'), (b"o<&+X'F4AQ8H,LU,N`&r", 10, b'BK5u.QHk1Driey7bvnFTH.', b'$2a$10$BK5u.QHk1Driey7bvnFTH.ESINe9YntUMcVgFDfkC.Vbhc9vMhNX2'), (b"o<&+X'F4AQ8H,LU,N`&r", 12, b'BK5u.QHk1Driey7bvnFTH.', b'$2a$12$BK5u.QHk1Driey7bvnFTH.QM1/nnGe/f5cTzb6XTTi/vMzcAnycqG')]
        for idx, (password, cost, salt64, result) in enumerate(tvs):
            x = bcrypt(password, cost, salt=_bcrypt_decode(salt64))
            self.assertEqual(x, result)
            bcrypt_check(password, result)

    def test_long_passwords(self):
        tvs = [(b'g*3Q45="8NNgpT&mbMJ$Omfr.#ZeW?FP=CE$#roHd?97uL0F-]`?u73c"\\[."*)qU34@VG', 4, b'T2XJ5MOWvHQZRijl8LIKkO', b'$2a$04$T2XJ5MOWvHQZRijl8LIKkOQKIyX75KBfuLsuRYOJz5OjwBNF2lM8a'), (b'\\M+*8;&QE=Ll[>5?Ui"^ai#iQH7ZFtNMfs3AROnIncE9"BNNoEgO[[*Yk8;RQ(#S,;I+aT', 5, b'wgkOlGNXIVE2fWkT3gyRoO', b'$2a$05$wgkOlGNXIVE2fWkT3gyRoOqWi4gbi1Wv2Q2Jx3xVs3apl1w.Wtj8C'), (b"M.E1=dt<.L0Q&p;94NfGm_Oo23+Kpl@M5?WIAL.[@/:'S)W96G8N^AWb7_smmC]>7#fGoB", 6, b'W9zTCl35nEvUukhhFzkKMe', b'$2a$06$W9zTCl35nEvUukhhFzkKMekjT9/pj7M0lihRVEZrX3m8/SBNZRX7i')]
        for idx, (password, cost, salt64, result) in enumerate(tvs):
            x = bcrypt(password, cost, salt=_bcrypt_decode(salt64))
            self.assertEqual(x, result)
            bcrypt_check(password, result)

    def test_increasing_password_length(self):
        tvs = [(b'a', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.l4WvgHIVg17ZawDIrDM2IjlE64GDNQS'), (b'aa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.AyUxBk.ThHlsLvRTH7IqcG7yVHJ3SXq'), (b'aaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.BxOVac5xPB6XFdRc/ZrzM9FgZkqmvbW'), (b'aaaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.Qbr209bpCtfl5hN7UQlG/L4xiD3AKau'), (b'aaaaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.oWszihPjDZI0ypReKsaDOW1jBl7oOii'), (b'aaaaaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ./k.Xxn9YiqtV/sxh3EHbnOHd0Qsq27K'), (b'aaaaaaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.PYJqRFQbgRbIjMd5VNKmdKS4sBVOyDe'), (b'aaaaaaaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ..VMYfzaw1wP/SGxowpLeGf13fxCCt.q'), (b'aaaaaaaaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.5B0p054nO5WgAD1n04XslDY/bqY9RJi'), (b'aaaaaaaaaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.INBTgqm7sdlBJDg.J5mLMSRK25ri04y'), (b'aaaaaaaaaaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.s3y7CdFD0OR5p6rsZw/eZ.Dla40KLfm'), (b'aaaaaaaaaaaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.Jx742Djra6Q7PqJWnTAS.85c28g.Siq'), (b'aaaaaaaaaaaaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.oKMXW3EZcPHcUV0ib5vDBnh9HojXnLu'), (b'aaaaaaaaaaaaaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.w6nIjWpDPNSH5pZUvLjC1q25ONEQpeS'), (b'aaaaaaaaaaaaaaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.k1b2/r9A/hxdwKEKurg6OCn4MwMdiGq'), (b'aaaaaaaaaaaaaaaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.3prCNHVX1Ws.7Hm2bJxFUnQOX9f7DFa')]
        for idx, (password, cost, salt64, result) in enumerate(tvs):
            x = bcrypt(password, cost, salt=_bcrypt_decode(salt64))
            self.assertEqual(x, result)
            bcrypt_check(password, result)

    def test_non_ascii_characters(self):
        tvs = [('àèìòùÀÈÌÒÙáéíóúýÁÉÍÓÚÝðÐ', 4, b'D3qS2aoTVyqM7z8v8crLm.', b'$2a$04$D3qS2aoTVyqM7z8v8crLm.3nKt4CzBZJbyFB.ZebmfCvRw7BGs.Xm'), ('àèìòùÀÈÌÒÙáéíóúýÁÉÍÓÚÝðÐ', 5, b'VA1FujiOCMPkUHQ8kF7IaO', b'$2a$05$VA1FujiOCMPkUHQ8kF7IaOg7NGaNvpxwWzSluQutxEVmbZItRTsAa'), ('àèìòùÀÈÌÒÙáéíóúýÁÉÍÓÚÝðÐ', 6, b'TXiaNrPeBSz5ugiQlehRt.', b'$2a$06$TXiaNrPeBSz5ugiQlehRt.gwpeDQnXWteQL4z2FulouBr6G7D9KUi'), ('âêîôûÂÊÎÔÛãñõÃÑÕäëïöüÿ', 4, b'YTn1Qlvps8e1odqMn6G5x.', b'$2a$04$YTn1Qlvps8e1odqMn6G5x.85pqKql6w773EZJAExk7/BatYAI4tyO'), ('âêîôûÂÊÎÔÛãñõÃÑÕäëïöüÿ', 5, b'C.8k5vJKD2NtfrRI9o17DO', b'$2a$05$C.8k5vJKD2NtfrRI9o17DOfIW0XnwItA529vJnh2jzYTb1QdoY0py'), ('âêîôûÂÊÎÔÛãñõÃÑÕäëïöüÿ', 6, b'xqfRPj3RYAgwurrhcA6uRO', b'$2a$06$xqfRPj3RYAgwurrhcA6uROtGlXDp/U6/gkoDYHwlubtcVcNft5.vW'), ('ÄËÏÖÜŸåÅæÆœŒßçÇøØ¢¿¡€', 4, b'y8vGgMmr9EdyxP9rmMKjH.', b'$2a$04$y8vGgMmr9EdyxP9rmMKjH.wv2y3r7yRD79gykQtmb3N3zrwjKsyay'), ('ÄËÏÖÜŸåÅæÆœŒßçÇøØ¢¿¡€', 5, b'iYH4XIKAOOm/xPQs7xKP1u', b'$2a$05$iYH4XIKAOOm/xPQs7xKP1upD0cWyMn3Jf0ZWiizXbEkVpS41K1dcO'), ('ÄËÏÖÜŸåÅæÆœŒßçÇøØ¢¿¡€', 6, b'wCOob.D0VV8twafNDB2ape', b'$2a$06$wCOob.D0VV8twafNDB2apegiGD5nqF6Y1e6K95q6Y.R8C4QGd265q'), ('ΔημοσιεύθηκεστηνΕφημερίδατης', 4, b'E5SQtS6P4568MDXW7cyUp.', b'$2a$04$E5SQtS6P4568MDXW7cyUp.18wfDisKZBxifnPZjAI1d/KTYMfHPYO'), ('АБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмН', 4, b'03e26gQFHhQwRNf81/ww9.', b'$2a$04$03e26gQFHhQwRNf81/ww9.p1UbrNwxpzWjLuT.zpTLH4t/w5WhAhC'), ('нОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮю', 4, b'PHNoJwpXCfe32nUtLv2Upu', b'$2a$04$PHNoJwpXCfe32nUtLv2UpuhJXOzd4k7IdFwnEpYwfJVCZ/f/.8Pje'), ('電电電島岛島兔兔兎龜龟亀國国国區区区', 4, b'wU4/0i1TmNl2u.1jIwBX.u', b'$2a$04$wU4/0i1TmNl2u.1jIwBX.uZUaOL3Rc5ID7nlQRloQh6q5wwhV/zLW'), ('诶比伊艾弗豆贝尔维吾艾尺开艾丝维贼德', 4, b'P4kreGLhCd26d4WIy7DJXu', b'$2a$04$P4kreGLhCd26d4WIy7DJXusPkhxLvBouzV6OXkL5EB0jux0osjsry')]
        for idx, (password, cost, salt64, result) in enumerate(tvs):
            x = bcrypt(password, cost, salt=_bcrypt_decode(salt64))
            self.assertEqual(x, result)
            bcrypt_check(password, result)

    def test_special_case_salt(self):
        tvs = [('-O_=*N!2JP', 4, b'......................', b'$2a$04$......................JjuKLOX9OOwo5PceZZXSkaLDvdmgb82'), ('7B[$Q<4b>U', 5, b'......................', b'$2a$05$......................DRiedDQZRL3xq5A5FL8y7/6NM8a2Y5W'), ('>d5-I_8^.h', 6, b'......................', b'$2a$06$......................5Mq1Ng8jgDY.uHNU4h5p/x6BedzNH2W'), (')V`/UM/]1t', 4, b'.OC/.OC/.OC/.OC/.OC/.O', b'$2a$04$.OC/.OC/.OC/.OC/.OC/.OQIvKRDAam.Hm5/IaV/.hc7P8gwwIbmi'), (':@t2.bWuH]', 5, b'.OC/.OC/.OC/.OC/.OC/.O', b'$2a$05$.OC/.OC/.OC/.OC/.OC/.ONDbUvdOchUiKmQORX6BlkPofa/QxW9e'), ('b(#KljF5s"', 6, b'.OC/.OC/.OC/.OC/.OC/.O', b'$2a$06$.OC/.OC/.OC/.OC/.OC/.OHfTd9e7svOu34vi1PCvOcAEq07ST7.K'), ('@3YaJ^Xs]*', 4, b'eGA.eGA.eGA.eGA.eGA.e.', b'$2a$04$eGA.eGA.eGA.eGA.eGA.e.stcmvh.R70m.0jbfSFVxlONdj1iws0C'), ('\'"5\\!k*C(p', 5, b'eGA.eGA.eGA.eGA.eGA.e.', b'$2a$05$eGA.eGA.eGA.eGA.eGA.e.vR37mVSbfdHwu.F0sNMvgn8oruQRghy'), ("edEu7C?$'W", 6, b'eGA.eGA.eGA.eGA.eGA.e.', b'$2a$06$eGA.eGA.eGA.eGA.eGA.e.tSq0FN8MWHQXJXNFnHTPQKtA.n2a..G'), ('N7dHmg\\PI^', 4, b'999999999999999999999u', b'$2a$04$999999999999999999999uCZfA/pLrlyngNDMq89r1uUk.bQ9icOu'), ('"eJuHh!)7*', 5, b'999999999999999999999u', b'$2a$05$999999999999999999999uj8Pfx.ufrJFAoWFLjapYBS5vVEQQ/hK'), ('ZeDRJ:_tu:', 6, b'999999999999999999999u', b'$2a$06$999999999999999999999u6RB0P9UmbdbQgjoQFEJsrvrKe.BoU6q')]
        for idx, (password, cost, salt64, result) in enumerate(tvs):
            x = bcrypt(password, cost, salt=_bcrypt_decode(salt64))
            self.assertEqual(x, result)
            bcrypt_check(password, result)