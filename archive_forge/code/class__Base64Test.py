from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
class _Base64Test(TestCase):
    """common tests for all Base64Engine instances"""
    engine = None
    encoded_data = None
    encoded_ints = None
    bad_byte = b'?'

    def m(self, *offsets):
        """generate byte string from offsets"""
        return join_bytes((self.engine.bytemap[o:o + 1] for o in offsets))

    def test_encode_bytes(self):
        """test encode_bytes() against reference inputs"""
        engine = self.engine
        encode = engine.encode_bytes
        for raw, encoded in self.encoded_data:
            result = encode(raw)
            self.assertEqual(result, encoded, 'encode %r:' % (raw,))

    def test_encode_bytes_bad(self):
        """test encode_bytes() with bad input"""
        engine = self.engine
        encode = engine.encode_bytes
        self.assertRaises(TypeError, encode, u('\x00'))
        self.assertRaises(TypeError, encode, None)

    def test_decode_bytes(self):
        """test decode_bytes() against reference inputs"""
        engine = self.engine
        decode = engine.decode_bytes
        for raw, encoded in self.encoded_data:
            result = decode(encoded)
            self.assertEqual(result, raw, 'decode %r:' % (encoded,))

    def test_decode_bytes_padding(self):
        """test decode_bytes() ignores padding bits"""
        bchr = (lambda v: bytes([v])) if PY3 else chr
        engine = self.engine
        m = self.m
        decode = engine.decode_bytes
        BNULL = b'\x00'
        self.assertEqual(decode(m(0, 0)), BNULL)
        for i in range(0, 6):
            if engine.big:
                correct = BNULL if i < 4 else bchr(1 << i - 4)
            else:
                correct = bchr(1 << i + 6) if i < 2 else BNULL
            self.assertEqual(decode(m(0, 1 << i)), correct, '%d/4 bits:' % i)
        self.assertEqual(decode(m(0, 0, 0)), BNULL * 2)
        for i in range(0, 6):
            if engine.big:
                correct = BNULL if i < 2 else bchr(1 << i - 2)
            else:
                correct = bchr(1 << i + 4) if i < 4 else BNULL
            self.assertEqual(decode(m(0, 0, 1 << i)), BNULL + correct, '%d/2 bits:' % i)

    def test_decode_bytes_bad(self):
        """test decode_bytes() with bad input"""
        engine = self.engine
        decode = engine.decode_bytes
        self.assertRaises(ValueError, decode, engine.bytemap[:5])
        self.assertTrue(self.bad_byte not in engine.bytemap)
        self.assertRaises(ValueError, decode, self.bad_byte * 4)
        self.assertRaises(TypeError, decode, engine.charmap[:4])
        self.assertRaises(TypeError, decode, None)

    def test_codec(self):
        """test encode_bytes/decode_bytes against random data"""
        engine = self.engine
        from passlib.utils import getrandbytes, getrandstr
        rng = self.getRandom()
        saw_zero = False
        for i in irange(500):
            size = rng.randint(1 if saw_zero else 0, 12)
            if not size:
                saw_zero = True
            enc_size = (4 * size + 2) // 3
            raw = getrandbytes(rng, size)
            encoded = engine.encode_bytes(raw)
            self.assertEqual(len(encoded), enc_size)
            result = engine.decode_bytes(encoded)
            self.assertEqual(result, raw)
            if size % 4 == 1:
                size += rng.choice([-1, 1, 2])
            raw_size = 3 * size // 4
            encoded = getrandstr(rng, engine.bytemap, size)
            raw = engine.decode_bytes(encoded)
            self.assertEqual(len(raw), raw_size, 'encoded %d:' % size)
            result = engine.encode_bytes(raw)
            if size % 4:
                self.assertEqual(result[:-1], encoded[:-1])
            else:
                self.assertEqual(result, encoded)

    def test_repair_unused(self):
        """test repair_unused()"""
        from passlib.utils import getrandstr
        rng = self.getRandom()
        engine = self.engine
        check_repair_unused = self.engine.check_repair_unused
        i = 0
        while i < 300:
            size = rng.randint(0, 23)
            cdata = getrandstr(rng, engine.charmap, size).encode('ascii')
            if size & 3 == 1:
                self.assertRaises(ValueError, check_repair_unused, cdata)
                continue
            rdata = engine.encode_bytes(engine.decode_bytes(cdata))
            if rng.random() < 0.5:
                cdata = cdata.decode('ascii')
                rdata = rdata.decode('ascii')
            if cdata == rdata:
                ok, result = check_repair_unused(cdata)
                self.assertFalse(ok)
                self.assertEqual(result, rdata)
            else:
                self.assertNotEqual(size % 4, 0)
                ok, result = check_repair_unused(cdata)
                self.assertTrue(ok)
                self.assertEqual(result, rdata)
            i += 1
    transposed = [(b'3"\x11', b'\x11"3', [2, 1, 0]), (b'"3\x11', b'\x11"3', [1, 2, 0])]
    transposed_dups = [(b'\x11\x11"', b'\x11"3', [0, 0, 1])]

    def test_encode_transposed_bytes(self):
        """test encode_transposed_bytes()"""
        engine = self.engine
        for result, input, offsets in self.transposed + self.transposed_dups:
            tmp = engine.encode_transposed_bytes(input, offsets)
            out = engine.decode_bytes(tmp)
            self.assertEqual(out, result)
        self.assertRaises(TypeError, engine.encode_transposed_bytes, u('a'), [])

    def test_decode_transposed_bytes(self):
        """test decode_transposed_bytes()"""
        engine = self.engine
        for input, result, offsets in self.transposed:
            tmp = engine.encode_bytes(input)
            out = engine.decode_transposed_bytes(tmp, offsets)
            self.assertEqual(out, result)

    def test_decode_transposed_bytes_bad(self):
        """test decode_transposed_bytes() fails if map is a one-way"""
        engine = self.engine
        for input, _, offsets in self.transposed_dups:
            tmp = engine.encode_bytes(input)
            self.assertRaises(TypeError, engine.decode_transposed_bytes, tmp, offsets)

    def check_int_pair(self, bits, encoded_pairs):
        """helper to check encode_intXX & decode_intXX functions"""
        rng = self.getRandom()
        engine = self.engine
        encode = getattr(engine, 'encode_int%s' % bits)
        decode = getattr(engine, 'decode_int%s' % bits)
        pad = -bits % 6
        chars = (bits + pad) // 6
        upper = 1 << bits
        for value, encoded in encoded_pairs:
            result = encode(value)
            self.assertIsInstance(result, bytes)
            self.assertEqual(result, encoded)
        self.assertRaises(ValueError, encode, -1)
        self.assertRaises(ValueError, encode, upper)
        for value, encoded in encoded_pairs:
            self.assertEqual(decode(encoded), value, 'encoded %r:' % (encoded,))
        m = self.m
        self.assertRaises(ValueError, decode, m(0) * (chars + 1))
        self.assertRaises(ValueError, decode, m(0) * (chars - 1))
        self.assertRaises(ValueError, decode, self.bad_byte * chars)
        self.assertRaises(TypeError, decode, engine.charmap[0])
        self.assertRaises(TypeError, decode, None)
        from passlib.utils import getrandstr
        for i in irange(100):
            value = rng.randint(0, upper - 1)
            encoded = encode(value)
            self.assertEqual(len(encoded), chars)
            self.assertEqual(decode(encoded), value)
            encoded = getrandstr(rng, engine.bytemap, chars)
            value = decode(encoded)
            self.assertGreaterEqual(value, 0, 'decode %r out of bounds:' % encoded)
            self.assertLess(value, upper, 'decode %r out of bounds:' % encoded)
            result = encode(value)
            if pad:
                self.assertEqual(result[:-2], encoded[:-2])
            else:
                self.assertEqual(result, encoded)

    def test_int6(self):
        engine = self.engine
        m = self.m
        self.check_int_pair(6, [(0, m(0)), (63, m(63))])

    def test_int12(self):
        engine = self.engine
        m = self.m
        self.check_int_pair(12, [(0, m(0, 0)), (63, m(0, 63) if engine.big else m(63, 0)), (4095, m(63, 63))])

    def test_int24(self):
        engine = self.engine
        m = self.m
        self.check_int_pair(24, [(0, m(0, 0, 0, 0)), (63, m(0, 0, 0, 63) if engine.big else m(63, 0, 0, 0)), (16777215, m(63, 63, 63, 63))])

    def test_int64(self):
        engine = self.engine
        m = self.m
        self.check_int_pair(64, [(0, m(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)), (63, m(0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 60) if engine.big else m(63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)), ((1 << 64) - 1, m(63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 60) if engine.big else m(63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 15))])

    def test_encoded_ints(self):
        """test against reference integer encodings"""
        if not self.encoded_ints:
            raise self.skipTests('none defined for class')
        engine = self.engine
        for data, value, bits in self.encoded_ints:
            encode = getattr(engine, 'encode_int%d' % bits)
            decode = getattr(engine, 'decode_int%d' % bits)
            self.assertEqual(encode(value), data)
            self.assertEqual(decode(data), value)