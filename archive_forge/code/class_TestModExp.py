import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Util.py3compat import *
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
from Cryptodome.Hash import SHAKE128
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Math._IntegerCustom import _raw_montgomery
from Cryptodome.Random.random import StrongRandom
class TestModExp(unittest.TestCase):

    def test_small(self):
        self.assertEqual(1, monty_pow(11, 12, 19))

    def test_large_1(self):
        base = 25711008708143844408671393477458601640355247900524685364822015
        expected = pow(base, exponent1, modulus1)
        result = monty_pow(base, exponent1, modulus1)
        self.assertEqual(result, expected)

    def test_zero_exp(self):
        base = 25711008708143844408671393477458601640355247900524685364822015
        result = monty_pow(base, 0, modulus1)
        self.assertEqual(result, 1)

    def test_zero_base(self):
        result = monty_pow(0, exponent1, modulus1)
        self.assertEqual(result, 0)

    def test_zero_modulus(self):
        base = 100433627766186892221372630771322662657637687111424552206335
        self.assertRaises(ExceptionModulus, monty_pow, base, exponent1, 0)
        self.assertRaises(ExceptionModulus, monty_pow, 0, 0, 0)

    def test_larger_exponent(self):
        base = modulus1 - 268435455
        expected = pow(base, modulus1 << 64, modulus1)
        result = monty_pow(base, modulus1 << 64, modulus1)
        self.assertEqual(result, expected)

    def test_even_modulus(self):
        base = modulus1 >> 4
        self.assertRaises(ExceptionModulus, monty_pow, base, exponent1, modulus1 - 1)

    def test_several_lengths(self):
        prng = SHAKE128.new().update(b('Test'))
        for length in range(1, 100):
            modulus2 = Integer.from_bytes(prng.read(length)) | 1
            base = Integer.from_bytes(prng.read(length)) % modulus2
            exponent2 = Integer.from_bytes(prng.read(length))
            expected = pow(base, exponent2, modulus2)
            result = monty_pow(base, exponent2, modulus2)
            self.assertEqual(result, expected)

    def test_variable_exponent(self):
        prng = create_rng(b('Test variable exponent'))
        for i in range(20):
            for j in range(7):
                modulus = prng.getrandbits(8 * 30) | 1
                base = prng.getrandbits(8 * 30) % modulus
                exponent = prng.getrandbits(i * 8 + j)
                expected = pow(base, exponent, modulus)
                result = monty_pow(base, exponent, modulus)
                self.assertEqual(result, expected)
                exponent ^= (1 << i * 8 + j) - 1
                expected = pow(base, exponent, modulus)
                result = monty_pow(base, exponent, modulus)
                self.assertEqual(result, expected)

    def test_stress_63(self):
        prng = create_rng(b('Test 63'))
        length = 63
        for _ in range(2000):
            modulus = prng.getrandbits(8 * length) | 1
            base = prng.getrandbits(8 * length) % modulus
            exponent = prng.getrandbits(8 * length)
            expected = pow(base, exponent, modulus)
            result = monty_pow(base, exponent, modulus)
            self.assertEqual(result, expected)

    def test_stress_64(self):
        prng = create_rng(b('Test 64'))
        length = 64
        for _ in range(2000):
            modulus = prng.getrandbits(8 * length) | 1
            base = prng.getrandbits(8 * length) % modulus
            exponent = prng.getrandbits(8 * length)
            expected = pow(base, exponent, modulus)
            result = monty_pow(base, exponent, modulus)
            self.assertEqual(result, expected)

    def test_stress_65(self):
        prng = create_rng(b('Test 65'))
        length = 65
        for _ in range(2000):
            modulus = prng.getrandbits(8 * length) | 1
            base = prng.getrandbits(8 * length) % modulus
            exponent = prng.getrandbits(8 * length)
            expected = pow(base, exponent, modulus)
            result = monty_pow(base, exponent, modulus)
            self.assertEqual(result, expected)