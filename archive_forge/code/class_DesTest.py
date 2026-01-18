from __future__ import with_statement, division
from functools import partial
from passlib.utils import getrandbytes
from passlib.tests.utils import TestCase
class DesTest(TestCase):
    descriptionPrefix = 'passlib.crypto.des'
    des_test_vectors = [(0, 0, 10134873677816210343), (18446744073709551615, 18446744073709551615, 8311870395893341272), (3458764513820540928, 1152921504606846977, 10776672327577195899), (1229782938247303441, 1229782938247303441, 17583031148182684979), (81985529216486895, 1229782938247303441, 1686191225890296621), (1229782938247303441, 81985529216486895, 9969529180854481629), (0, 0, 10134873677816210343), (18364758544493064720, 81985529216486895, 17093932802483993796), (8980477021735513687, 117611255094011714, 7570369612612015003), (86088881178490734, 6689337107006052314, 8806961764262204017), (549741787767056006, 164614723499094386, 9695893007742818714), (4055886516176695710, 5856169732564009994, 8176434030039898922), (340327136592049590, 4827089350059065250, 12625836341772173461), (77609513531011790, 404019981405066298, 9702267560690899035), (103848277426812390, 528848464848052690, 924322050668092425), (4839539656546808830, 8513233451820730474, 16890586767283661690), (551430852305365526, 4313623329492117506, 16129161034940226063), (317663223366892335, 2780233246153072794, 6652164766532288648), (4021832892757538118, 1607044272340030002, 732660321565846391), (2236079052714821214, 7711690988273491146, 17229628950091290458), (6359121586699264374, 21346945391353954, 9853609588653157974), (168909270948622343, 5191868619451491058, 11671519704656251734), (5294331816167286159, 4860862602324950266, 8052186200196056406), (5742192969548264359, 517143888688272018, 3396528426238910892), (5325890758360836543, 215703803915661610, 6515408130789920330), (108949354149783254, 2133963297529473218, 6866867443762671169), (2042522189576687599, 3482745036057028954, 7204282554404960147), (72340172838076673, 81985529216486895, 7024271870936510720), (2242545357694045710, 81985529216486895, 15822700226042971654), (16212643094166696446, 81985529216486895, 17131642157689064647), (0, 18446744073709551615, 3843066582818235473), (18446744073709551615, 0, 14603677490891316142), (81985529216486895, 0, 15408028147960528141), (18364758544493064720, 18446744073709551615, 3038715925749023474)]

    def test_01_expand(self):
        """expand_des_key()"""
        from passlib.crypto.des import expand_des_key, shrink_des_key, _KDATA_MASK, INT_56_MASK
        for key1, _, _ in self.des_test_vectors:
            key2 = shrink_des_key(key1)
            key3 = expand_des_key(key2)
            self.assertEqual(key3, key1 & _KDATA_MASK)
        self.assertRaises(TypeError, expand_des_key, 1.0)
        self.assertRaises(ValueError, expand_des_key, INT_56_MASK + 1)
        self.assertRaises(ValueError, expand_des_key, b'\x00' * 8)
        self.assertRaises(ValueError, expand_des_key, -1)
        self.assertRaises(ValueError, expand_des_key, b'\x00' * 6)

    def test_02_shrink(self):
        """shrink_des_key()"""
        from passlib.crypto.des import expand_des_key, shrink_des_key, INT_64_MASK
        rng = self.getRandom()
        for i in range(20):
            key1 = getrandbytes(rng, 7)
            key2 = expand_des_key(key1)
            key3 = shrink_des_key(key2)
            self.assertEqual(key3, key1)
        self.assertRaises(TypeError, shrink_des_key, 1.0)
        self.assertRaises(ValueError, shrink_des_key, INT_64_MASK + 1)
        self.assertRaises(ValueError, shrink_des_key, b'\x00' * 9)
        self.assertRaises(ValueError, shrink_des_key, -1)
        self.assertRaises(ValueError, shrink_des_key, b'\x00' * 7)

    def _random_parity(self, key):
        """randomize parity bits"""
        from passlib.crypto.des import _KDATA_MASK, _KPARITY_MASK, INT_64_MASK
        rng = self.getRandom()
        return key & _KDATA_MASK | rng.randint(0, INT_64_MASK) & _KPARITY_MASK

    def test_03_encrypt_bytes(self):
        """des_encrypt_block()"""
        from passlib.crypto.des import des_encrypt_block, shrink_des_key, _pack64, _unpack64
        for key, plaintext, correct in self.des_test_vectors:
            key = _pack64(key)
            plaintext = _pack64(plaintext)
            correct = _pack64(correct)
            result = des_encrypt_block(key, plaintext)
            self.assertEqual(result, correct, 'key=%r plaintext=%r:' % (key, plaintext))
            key2 = shrink_des_key(key)
            result = des_encrypt_block(key2, plaintext)
            self.assertEqual(result, correct, 'key=%r shrink(key)=%r plaintext=%r:' % (key, key2, plaintext))
            for _ in range(20):
                key3 = _pack64(self._random_parity(_unpack64(key)))
                result = des_encrypt_block(key3, plaintext)
                self.assertEqual(result, correct, 'key=%r rndparity(key)=%r plaintext=%r:' % (key, key3, plaintext))
        stub = b'\x00' * 8
        self.assertRaises(TypeError, des_encrypt_block, 0, stub)
        self.assertRaises(ValueError, des_encrypt_block, b'\x00' * 6, stub)
        self.assertRaises(TypeError, des_encrypt_block, stub, 0)
        self.assertRaises(ValueError, des_encrypt_block, stub, b'\x00' * 7)
        self.assertRaises(ValueError, des_encrypt_block, stub, stub, salt=-1)
        self.assertRaises(ValueError, des_encrypt_block, stub, stub, salt=1 << 24)
        self.assertRaises(ValueError, des_encrypt_block, stub, stub, 0, rounds=0)

    def test_04_encrypt_ints(self):
        """des_encrypt_int_block()"""
        from passlib.crypto.des import des_encrypt_int_block
        for key, plaintext, correct in self.des_test_vectors:
            result = des_encrypt_int_block(key, plaintext)
            self.assertEqual(result, correct, 'key=%r plaintext=%r:' % (key, plaintext))
            for _ in range(20):
                key3 = self._random_parity(key)
                result = des_encrypt_int_block(key3, plaintext)
                self.assertEqual(result, correct, 'key=%r rndparity(key)=%r plaintext=%r:' % (key, key3, plaintext))
        self.assertRaises(TypeError, des_encrypt_int_block, b'\x00', 0)
        self.assertRaises(ValueError, des_encrypt_int_block, -1, 0)
        self.assertRaises(TypeError, des_encrypt_int_block, 0, b'\x00')
        self.assertRaises(ValueError, des_encrypt_int_block, 0, -1)
        self.assertRaises(ValueError, des_encrypt_int_block, 0, 0, salt=-1)
        self.assertRaises(ValueError, des_encrypt_int_block, 0, 0, salt=1 << 24)
        self.assertRaises(ValueError, des_encrypt_int_block, 0, 0, 0, rounds=0)