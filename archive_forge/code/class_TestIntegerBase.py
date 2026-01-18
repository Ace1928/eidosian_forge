import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
class TestIntegerBase(unittest.TestCase):

    def setUp(self):
        raise NotImplementedError('To be implemented')

    def Integers(self, *arg):
        return map(self.Integer, arg)

    def test_init_and_equality(self):
        Integer = self.Integer
        v1 = Integer(23)
        v2 = Integer(v1)
        v3 = Integer(-9)
        self.assertRaises(ValueError, Integer, 1.0)
        v4 = Integer(10 ** 10)
        v5 = Integer(-10 ** 10)
        v6 = Integer(65535)
        v7 = Integer(4294967295)
        v8 = Integer(18446744073709551615)
        self.assertEqual(v1, v1)
        self.assertEqual(v1, 23)
        self.assertEqual(v1, v2)
        self.assertEqual(v3, -9)
        self.assertEqual(v4, 10 ** 10)
        self.assertEqual(v5, -10 ** 10)
        self.assertEqual(v6, 65535)
        self.assertEqual(v7, 4294967295)
        self.assertEqual(v8, 18446744073709551615)
        self.assertFalse(v1 == v4)
        v6 = Integer(v1)
        self.assertEqual(v1, v6)
        self.assertFalse(Integer(0) == None)

    def test_conversion_to_int(self):
        v1, v2 = self.Integers(-23, 2 ** 1000)
        self.assertEqual(int(v1), -23)
        self.assertEqual(int(v2), 2 ** 1000)

    def test_equality_with_ints(self):
        v1, v2, v3 = self.Integers(23, -89, 2 ** 1000)
        self.assertTrue(v1 == 23)
        self.assertTrue(v2 == -89)
        self.assertFalse(v1 == 24)
        self.assertTrue(v3 == 2 ** 1000)

    def test_conversion_to_str(self):
        v1, v2, v3, v4 = self.Integers(20, 0, -20, 2 ** 1000)
        self.assertTrue(str(v1) == '20')
        self.assertTrue(str(v2) == '0')
        self.assertTrue(str(v3) == '-20')
        self.assertTrue(str(v4) == '10715086071862673209484250490600018105614048117055336074437503883703510511249361224931983788156958581275946729175531468251871452856923140435984577574698574803934567774824230985421074605062371141877954182153046474983581941267398767559165543946077062914571196477686542167660429831652624386837205668069376')

    def test_repr(self):
        v1, v2 = self.Integers(-1, 2 ** 80)
        self.assertEqual(repr(v1), 'Integer(-1)')
        self.assertEqual(repr(v2), 'Integer(1208925819614629174706176)')

    def test_conversion_to_bytes(self):
        Integer = self.Integer
        v1 = Integer(23)
        self.assertEqual(b('\x17'), v1.to_bytes())
        v2 = Integer(65534)
        self.assertEqual(b('ÿþ'), v2.to_bytes())
        self.assertEqual(b('\x00ÿþ'), v2.to_bytes(3))
        self.assertRaises(ValueError, v2.to_bytes, 1)
        self.assertEqual(b('þÿ'), v2.to_bytes(byteorder='little'))
        self.assertEqual(b('þÿ\x00'), v2.to_bytes(3, byteorder='little'))
        v3 = Integer(-90)
        self.assertRaises(ValueError, v3.to_bytes)
        self.assertRaises(ValueError, v3.to_bytes, byteorder='bittle')

    def test_conversion_from_bytes(self):
        Integer = self.Integer
        v1 = Integer.from_bytes(b'\x00')
        self.assertTrue(isinstance(v1, Integer))
        self.assertEqual(0, v1)
        v2 = Integer.from_bytes(b'\x00\x01')
        self.assertEqual(1, v2)
        v3 = Integer.from_bytes(b'\xff\xff')
        self.assertEqual(65535, v3)
        v4 = Integer.from_bytes(b'\x00\x01', 'big')
        self.assertEqual(1, v4)
        v5 = Integer.from_bytes(b'\x00\x01', byteorder='big')
        self.assertEqual(1, v5)
        v6 = Integer.from_bytes(b'\x00\x01', byteorder='little')
        self.assertEqual(256, v6)
        self.assertRaises(ValueError, Integer.from_bytes, b'\t', 'bittle')

    def test_inequality(self):
        v1, v2, v3, v4 = self.Integers(89, 89, 90, -8)
        self.assertTrue(v1 != v3)
        self.assertTrue(v1 != 90)
        self.assertFalse(v1 != v2)
        self.assertFalse(v1 != 89)
        self.assertTrue(v1 != v4)
        self.assertTrue(v4 != v1)
        self.assertTrue(self.Integer(0) != None)

    def test_less_than(self):
        v1, v2, v3, v4, v5 = self.Integers(13, 13, 14, -8, 2 ** 10)
        self.assertTrue(v1 < v3)
        self.assertTrue(v1 < 14)
        self.assertFalse(v1 < v2)
        self.assertFalse(v1 < 13)
        self.assertTrue(v4 < v1)
        self.assertFalse(v1 < v4)
        self.assertTrue(v1 < v5)
        self.assertFalse(v5 < v1)

    def test_less_than_or_equal(self):
        v1, v2, v3, v4, v5 = self.Integers(13, 13, 14, -4, 2 ** 10)
        self.assertTrue(v1 <= v1)
        self.assertTrue(v1 <= 13)
        self.assertTrue(v1 <= v2)
        self.assertTrue(v1 <= 14)
        self.assertTrue(v1 <= v3)
        self.assertFalse(v1 <= v4)
        self.assertTrue(v1 <= v5)
        self.assertFalse(v5 <= v1)

    def test_more_than(self):
        v1, v2, v3, v4, v5 = self.Integers(13, 13, 14, -8, 2 ** 10)
        self.assertTrue(v3 > v1)
        self.assertTrue(v3 > 13)
        self.assertFalse(v1 > v1)
        self.assertFalse(v1 > v2)
        self.assertFalse(v1 > 13)
        self.assertTrue(v1 > v4)
        self.assertFalse(v4 > v1)
        self.assertTrue(v5 > v1)
        self.assertFalse(v1 > v5)

    def test_more_than_or_equal(self):
        v1, v2, v3, v4 = self.Integers(13, 13, 14, -4)
        self.assertTrue(v3 >= v1)
        self.assertTrue(v3 >= 13)
        self.assertTrue(v1 >= v2)
        self.assertTrue(v1 >= v1)
        self.assertTrue(v1 >= 13)
        self.assertFalse(v4 >= v1)

    def test_bool(self):
        v1, v2, v3, v4 = self.Integers(0, 10, -9, 2 ** 10)
        self.assertFalse(v1)
        self.assertFalse(bool(v1))
        self.assertTrue(v2)
        self.assertTrue(bool(v2))
        self.assertTrue(v3)
        self.assertTrue(v4)

    def test_is_negative(self):
        v1, v2, v3, v4, v5 = self.Integers(-3 ** 100, -3, 0, 3, 3 ** 100)
        self.assertTrue(v1.is_negative())
        self.assertTrue(v2.is_negative())
        self.assertFalse(v4.is_negative())
        self.assertFalse(v5.is_negative())

    def test_addition(self):
        v1, v2, v3 = self.Integers(7, 90, -7)
        self.assertTrue(isinstance(v1 + v2, self.Integer))
        self.assertEqual(v1 + v2, 97)
        self.assertEqual(v1 + 90, 97)
        self.assertEqual(v1 + v3, 0)
        self.assertEqual(v1 + -7, 0)
        self.assertEqual(v1 + 2 ** 10, 2 ** 10 + 7)

    def test_subtraction(self):
        v1, v2, v3 = self.Integers(7, 90, -7)
        self.assertTrue(isinstance(v1 - v2, self.Integer))
        self.assertEqual(v2 - v1, 83)
        self.assertEqual(v2 - 7, 83)
        self.assertEqual(v2 - v3, 97)
        self.assertEqual(v1 - -7, 14)
        self.assertEqual(v1 - 2 ** 10, 7 - 2 ** 10)

    def test_multiplication(self):
        v1, v2, v3, v4 = self.Integers(4, 5, -2, 2 ** 10)
        self.assertTrue(isinstance(v1 * v2, self.Integer))
        self.assertEqual(v1 * v2, 20)
        self.assertEqual(v1 * 5, 20)
        self.assertEqual(v1 * -2, -8)
        self.assertEqual(v1 * 2 ** 10, 4 * 2 ** 10)

    def test_floor_div(self):
        v1, v2, v3 = self.Integers(3, 8, 2 ** 80)
        self.assertTrue(isinstance(v1 // v2, self.Integer))
        self.assertEqual(v2 // v1, 2)
        self.assertEqual(v2 // 3, 2)
        self.assertEqual(v2 // -3, -3)
        self.assertEqual(v3 // 2 ** 79, 2)
        self.assertRaises(ZeroDivisionError, lambda: v1 // 0)

    def test_remainder(self):
        v1, v2, v3 = self.Integers(23, 5, -4)
        self.assertTrue(isinstance(v1 % v2, self.Integer))
        self.assertEqual(v1 % v2, 3)
        self.assertEqual(v1 % 5, 3)
        self.assertEqual(v3 % 5, 1)
        self.assertEqual(v1 % 2 ** 10, 23)
        self.assertRaises(ZeroDivisionError, lambda: v1 % 0)
        self.assertRaises(ValueError, lambda: v1 % -6)

    def test_simple_exponentiation(self):
        v1, v2, v3 = self.Integers(4, 3, -2)
        self.assertTrue(isinstance(v1 ** v2, self.Integer))
        self.assertEqual(v1 ** v2, 64)
        self.assertEqual(pow(v1, v2), 64)
        self.assertEqual(v1 ** 3, 64)
        self.assertEqual(pow(v1, 3), 64)
        self.assertEqual(v3 ** 2, 4)
        self.assertEqual(v3 ** 3, -8)
        self.assertRaises(ValueError, pow, v1, -3)

    def test_modular_exponentiation(self):
        v1, v2, v3 = self.Integers(23, 5, 17)
        self.assertTrue(isinstance(pow(v1, v2, v3), self.Integer))
        self.assertEqual(pow(v1, v2, v3), 7)
        self.assertEqual(pow(v1, 5, v3), 7)
        self.assertEqual(pow(v1, v2, 17), 7)
        self.assertEqual(pow(v1, 5, 17), 7)
        self.assertEqual(pow(v1, 0, 17), 1)
        self.assertEqual(pow(v1, 1, 2 ** 80), 23)
        self.assertEqual(pow(v1, 2 ** 80, 89298), 17689)
        self.assertRaises(ZeroDivisionError, pow, v1, 5, 0)
        self.assertRaises(ValueError, pow, v1, 5, -4)
        self.assertRaises(ValueError, pow, v1, -3, 8)

    def test_inplace_exponentiation(self):
        v1 = self.Integer(4)
        v1.inplace_pow(2)
        self.assertEqual(v1, 16)
        v1 = self.Integer(4)
        v1.inplace_pow(2, 15)
        self.assertEqual(v1, 1)

    def test_abs(self):
        v1, v2, v3, v4, v5 = self.Integers(-2 ** 100, -2, 0, 2, 2 ** 100)
        self.assertEqual(abs(v1), 2 ** 100)
        self.assertEqual(abs(v2), 2)
        self.assertEqual(abs(v3), 0)
        self.assertEqual(abs(v4), 2)
        self.assertEqual(abs(v5), 2 ** 100)

    def test_sqrt(self):
        v1, v2, v3, v4 = self.Integers(-2, 0, 49, 10 ** 100)
        self.assertRaises(ValueError, v1.sqrt)
        self.assertEqual(v2.sqrt(), 0)
        self.assertEqual(v3.sqrt(), 7)
        self.assertEqual(v4.sqrt(), 10 ** 50)

    def test_sqrt_module(self):
        self.assertRaises(ValueError, self.Integer(5).sqrt, 0)
        self.assertRaises(ValueError, self.Integer(5).sqrt, -1)
        assert self.Integer(0).sqrt(5) == 0
        assert self.Integer(1).sqrt(5) in (1, 4)
        for p in (11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53):
            for i in range(0, p):
                square = i ** 2 % p
                res = self.Integer(square).sqrt(p)
                assert res in (i, p - i)
        self.assertRaises(ValueError, self.Integer(2).sqrt, 11)
        self.assertRaises(ValueError, self.Integer(4).sqrt, 10)
        assert self.Integer(5 - 11).sqrt(11) in (4, 7)
        assert self.Integer(5 + 11).sqrt(11) in (4, 7)

    def test_in_place_add(self):
        v1, v2 = self.Integers(10, 20)
        v1 += v2
        self.assertEqual(v1, 30)
        v1 += 10
        self.assertEqual(v1, 40)
        v1 += -1
        self.assertEqual(v1, 39)
        v1 += 2 ** 1000
        self.assertEqual(v1, 39 + 2 ** 1000)

    def test_in_place_sub(self):
        v1, v2 = self.Integers(10, 20)
        v1 -= v2
        self.assertEqual(v1, -10)
        v1 -= -100
        self.assertEqual(v1, 90)
        v1 -= 90000
        self.assertEqual(v1, -89910)
        v1 -= -100000
        self.assertEqual(v1, 10090)

    def test_in_place_mul(self):
        v1, v2 = self.Integers(3, 5)
        v1 *= v2
        self.assertEqual(v1, 15)
        v1 *= 2
        self.assertEqual(v1, 30)
        v1 *= -2
        self.assertEqual(v1, -60)
        v1 *= 2 ** 1000
        self.assertEqual(v1, -60 * 2 ** 1000)

    def test_in_place_modulus(self):
        v1, v2 = self.Integers(20, 7)
        v1 %= v2
        self.assertEqual(v1, 6)
        v1 %= 2 ** 1000
        self.assertEqual(v1, 6)
        v1 %= 2
        self.assertEqual(v1, 0)

        def t():
            v3 = self.Integer(9)
            v3 %= 0
        self.assertRaises(ZeroDivisionError, t)

    def test_and(self):
        v1, v2, v3 = self.Integers(244, 49, -15)
        self.assertTrue(isinstance(v1 & v2, self.Integer))
        self.assertEqual(v1 & v2, 48)
        self.assertEqual(v1 & 49, 48)
        self.assertEqual(v1 & v3, 240)
        self.assertEqual(v1 & -15, 240)
        self.assertEqual(v3 & -15, -15)
        self.assertEqual(v2 & 2 ** 1000 + 49, 49)

    def test_or(self):
        v1, v2, v3 = self.Integers(64, 130, -15)
        self.assertTrue(isinstance(v1 | v2, self.Integer))
        self.assertEqual(v1 | v2, 194)
        self.assertEqual(v1 | 130, 194)
        self.assertEqual(v2 | v3, -13)
        self.assertEqual(v2 | 2 ** 1000, 2 ** 1000 + 130)

    def test_right_shift(self):
        v1, v2, v3 = self.Integers(16, 1, -16)
        self.assertEqual(v1 >> 0, v1)
        self.assertTrue(isinstance(v1 >> v2, self.Integer))
        self.assertEqual(v1 >> v2, 8)
        self.assertEqual(v1 >> 1, 8)
        self.assertRaises(ValueError, lambda: v1 >> -1)
        self.assertEqual(v1 >> 2 ** 1000, 0)
        self.assertEqual(v3 >> 1, -8)
        self.assertEqual(v3 >> 2 ** 1000, -1)

    def test_in_place_right_shift(self):
        v1, v2, v3 = self.Integers(16, 1, -16)
        v1 >>= 0
        self.assertEqual(v1, 16)
        v1 >>= 1
        self.assertEqual(v1, 8)
        v1 >>= v2
        self.assertEqual(v1, 4)
        v3 >>= 1
        self.assertEqual(v3, -8)

        def l():
            v4 = self.Integer(144)
            v4 >>= -1
        self.assertRaises(ValueError, l)

        def m1():
            v4 = self.Integer(144)
            v4 >>= 2 ** 1000
            return v4
        self.assertEqual(0, m1())

        def m2():
            v4 = self.Integer(-1)
            v4 >>= 2 ** 1000
            return v4
        self.assertEqual(-1, m2())

    def _test_left_shift(self):
        v1, v2, v3 = self.Integers(16, 1, -16)
        self.assertEqual(v1 << 0, v1)
        self.assertTrue(isinstance(v1 << v2, self.Integer))
        self.assertEqual(v1 << v2, 32)
        self.assertEqual(v1 << 1, 32)
        self.assertEqual(v3 << 1, -32)
        self.assertRaises(ValueError, lambda: v1 << -1)
        self.assertRaises(ValueError, lambda: v1 << 2 ** 1000)

    def test_in_place_left_shift(self):
        v1, v2, v3 = self.Integers(16, 1, -16)
        v1 <<= 0
        self.assertEqual(v1, 16)
        v1 <<= 1
        self.assertEqual(v1, 32)
        v1 <<= v2
        self.assertEqual(v1, 64)
        v3 <<= 1
        self.assertEqual(v3, -32)

        def l():
            v4 = self.Integer(144)
            v4 <<= -1
        self.assertRaises(ValueError, l)

        def m():
            v4 = self.Integer(144)
            v4 <<= 2 ** 1000
        self.assertRaises(ValueError, m)

    def test_get_bit(self):
        v1, v2, v3 = self.Integers(258, -3, 1)
        self.assertEqual(v1.get_bit(0), 0)
        self.assertEqual(v1.get_bit(1), 1)
        self.assertEqual(v1.get_bit(v3), 1)
        self.assertEqual(v1.get_bit(8), 1)
        self.assertEqual(v1.get_bit(9), 0)
        self.assertRaises(ValueError, v1.get_bit, -1)
        self.assertEqual(v1.get_bit(2 ** 1000), 0)
        self.assertRaises(ValueError, v2.get_bit, -1)
        self.assertRaises(ValueError, v2.get_bit, 0)
        self.assertRaises(ValueError, v2.get_bit, 1)
        self.assertRaises(ValueError, v2.get_bit, 2 * 1000)

    def test_odd_even(self):
        v1, v2, v3, v4, v5 = self.Integers(0, 4, 17, -4, -17)
        self.assertTrue(v1.is_even())
        self.assertTrue(v2.is_even())
        self.assertFalse(v3.is_even())
        self.assertTrue(v4.is_even())
        self.assertFalse(v5.is_even())
        self.assertFalse(v1.is_odd())
        self.assertFalse(v2.is_odd())
        self.assertTrue(v3.is_odd())
        self.assertFalse(v4.is_odd())
        self.assertTrue(v5.is_odd())

    def test_size_in_bits(self):
        v1, v2, v3, v4 = self.Integers(0, 1, 256, -90)
        self.assertEqual(v1.size_in_bits(), 1)
        self.assertEqual(v2.size_in_bits(), 1)
        self.assertEqual(v3.size_in_bits(), 9)
        self.assertRaises(ValueError, v4.size_in_bits)

    def test_size_in_bytes(self):
        v1, v2, v3, v4, v5, v6 = self.Integers(0, 1, 255, 511, 65536, -9)
        self.assertEqual(v1.size_in_bytes(), 1)
        self.assertEqual(v2.size_in_bytes(), 1)
        self.assertEqual(v3.size_in_bytes(), 1)
        self.assertEqual(v4.size_in_bytes(), 2)
        self.assertEqual(v5.size_in_bytes(), 3)
        self.assertRaises(ValueError, v6.size_in_bits)

    def test_perfect_square(self):
        self.assertFalse(self.Integer(-9).is_perfect_square())
        self.assertTrue(self.Integer(0).is_perfect_square())
        self.assertTrue(self.Integer(1).is_perfect_square())
        self.assertFalse(self.Integer(2).is_perfect_square())
        self.assertFalse(self.Integer(3).is_perfect_square())
        self.assertTrue(self.Integer(4).is_perfect_square())
        self.assertTrue(self.Integer(39 * 39).is_perfect_square())
        self.assertFalse(self.Integer(39 * 39 + 1).is_perfect_square())
        for x in range(100, 1000):
            self.assertFalse(self.Integer(x ** 2 + 1).is_perfect_square())
            self.assertTrue(self.Integer(x ** 2).is_perfect_square())

    def test_fail_if_divisible_by(self):
        v1, v2, v3 = self.Integers(12, -12, 4)
        v1.fail_if_divisible_by(7)
        v2.fail_if_divisible_by(7)
        v2.fail_if_divisible_by(2 ** 80)
        self.assertRaises(ValueError, v1.fail_if_divisible_by, 4)
        self.assertRaises(ValueError, v1.fail_if_divisible_by, v3)

    def test_multiply_accumulate(self):
        v1, v2, v3 = self.Integers(4, 3, 2)
        v1.multiply_accumulate(v2, v3)
        self.assertEqual(v1, 10)
        v1.multiply_accumulate(v2, 2)
        self.assertEqual(v1, 16)
        v1.multiply_accumulate(3, v3)
        self.assertEqual(v1, 22)
        v1.multiply_accumulate(1, -2)
        self.assertEqual(v1, 20)
        v1.multiply_accumulate(-2, 1)
        self.assertEqual(v1, 18)
        v1.multiply_accumulate(1, 2 ** 1000)
        self.assertEqual(v1, 18 + 2 ** 1000)
        v1.multiply_accumulate(2 ** 1000, 1)
        self.assertEqual(v1, 18 + 2 ** 1001)

    def test_set(self):
        v1, v2 = self.Integers(3, 6)
        v1.set(v2)
        self.assertEqual(v1, 6)
        v1.set(9)
        self.assertEqual(v1, 9)
        v1.set(-2)
        self.assertEqual(v1, -2)
        v1.set(2 ** 1000)
        self.assertEqual(v1, 2 ** 1000)

    def test_inverse(self):
        v1, v2, v3, v4, v5, v6 = self.Integers(2, 5, -3, 0, 723872, 3433)
        self.assertTrue(isinstance(v1.inverse(v2), self.Integer))
        self.assertEqual(v1.inverse(v2), 3)
        self.assertEqual(v1.inverse(5), 3)
        self.assertEqual(v3.inverse(5), 3)
        self.assertEqual(v5.inverse(92929921), 58610507)
        self.assertEqual(v6.inverse(9912), 5353)
        self.assertRaises(ValueError, v2.inverse, 10)
        self.assertRaises(ValueError, v1.inverse, -3)
        self.assertRaises(ValueError, v4.inverse, 10)
        self.assertRaises(ZeroDivisionError, v2.inverse, 0)

    def test_inplace_inverse(self):
        v1, v2 = self.Integers(2, 5)
        v1.inplace_inverse(v2)
        self.assertEqual(v1, 3)

    def test_gcd(self):
        v1, v2, v3, v4 = self.Integers(6, 10, 17, -2)
        self.assertTrue(isinstance(v1.gcd(v2), self.Integer))
        self.assertEqual(v1.gcd(v2), 2)
        self.assertEqual(v1.gcd(10), 2)
        self.assertEqual(v1.gcd(v3), 1)
        self.assertEqual(v1.gcd(-2), 2)
        self.assertEqual(v4.gcd(6), 2)

    def test_lcm(self):
        v1, v2, v3, v4, v5 = self.Integers(6, 10, 17, -2, 0)
        self.assertTrue(isinstance(v1.lcm(v2), self.Integer))
        self.assertEqual(v1.lcm(v2), 30)
        self.assertEqual(v1.lcm(10), 30)
        self.assertEqual(v1.lcm(v3), 102)
        self.assertEqual(v1.lcm(-2), 6)
        self.assertEqual(v4.lcm(6), 6)
        self.assertEqual(v1.lcm(0), 0)
        self.assertEqual(v5.lcm(0), 0)

    def test_jacobi_symbol(self):
        data = ((1001, 1, 1), (19, 45, 1), (8, 21, -1), (5, 21, 1), (610, 987, -1), (1001, 9907, -1), (5, 3439601197, -1))
        js = self.Integer.jacobi_symbol
        for k in range(1, 30):
            self.assertEqual(js(k, 1), 1)
        for n in range(1, 30, 2):
            self.assertEqual(js(1, n), 1)
        self.assertRaises(ValueError, js, 6, -2)
        self.assertRaises(ValueError, js, 6, -1)
        self.assertRaises(ValueError, js, 6, 0)
        self.assertRaises(ValueError, js, 0, 0)
        self.assertRaises(ValueError, js, 6, 2)
        self.assertRaises(ValueError, js, 6, 4)
        self.assertRaises(ValueError, js, 6, 6)
        self.assertRaises(ValueError, js, 6, 8)
        for tv in data:
            self.assertEqual(js(tv[0], tv[1]), tv[2])
            self.assertEqual(js(self.Integer(tv[0]), tv[1]), tv[2])
            self.assertEqual(js(tv[0], self.Integer(tv[1])), tv[2])

    def test_jacobi_symbol_wikipedia(self):
        tv = [(3, [(1, 1), (2, -1), (3, 0), (4, 1), (5, -1), (6, 0), (7, 1), (8, -1), (9, 0), (10, 1), (11, -1), (12, 0), (13, 1), (14, -1), (15, 0), (16, 1), (17, -1), (18, 0), (19, 1), (20, -1), (21, 0), (22, 1), (23, -1), (24, 0), (25, 1), (26, -1), (27, 0), (28, 1), (29, -1), (30, 0)]), (5, [(1, 1), (2, -1), (3, -1), (4, 1), (5, 0), (6, 1), (7, -1), (8, -1), (9, 1), (10, 0), (11, 1), (12, -1), (13, -1), (14, 1), (15, 0), (16, 1), (17, -1), (18, -1), (19, 1), (20, 0), (21, 1), (22, -1), (23, -1), (24, 1), (25, 0), (26, 1), (27, -1), (28, -1), (29, 1), (30, 0)]), (7, [(1, 1), (2, 1), (3, -1), (4, 1), (5, -1), (6, -1), (7, 0), (8, 1), (9, 1), (10, -1), (11, 1), (12, -1), (13, -1), (14, 0), (15, 1), (16, 1), (17, -1), (18, 1), (19, -1), (20, -1), (21, 0), (22, 1), (23, 1), (24, -1), (25, 1), (26, -1), (27, -1), (28, 0), (29, 1), (30, 1)]), (9, [(1, 1), (2, 1), (3, 0), (4, 1), (5, 1), (6, 0), (7, 1), (8, 1), (9, 0), (10, 1), (11, 1), (12, 0), (13, 1), (14, 1), (15, 0), (16, 1), (17, 1), (18, 0), (19, 1), (20, 1), (21, 0), (22, 1), (23, 1), (24, 0), (25, 1), (26, 1), (27, 0), (28, 1), (29, 1), (30, 0)]), (11, [(1, 1), (2, -1), (3, 1), (4, 1), (5, 1), (6, -1), (7, -1), (8, -1), (9, 1), (10, -1), (11, 0), (12, 1), (13, -1), (14, 1), (15, 1), (16, 1), (17, -1), (18, -1), (19, -1), (20, 1), (21, -1), (22, 0), (23, 1), (24, -1), (25, 1), (26, 1), (27, 1), (28, -1), (29, -1), (30, -1)]), (13, [(1, 1), (2, -1), (3, 1), (4, 1), (5, -1), (6, -1), (7, -1), (8, -1), (9, 1), (10, 1), (11, -1), (12, 1), (13, 0), (14, 1), (15, -1), (16, 1), (17, 1), (18, -1), (19, -1), (20, -1), (21, -1), (22, 1), (23, 1), (24, -1), (25, 1), (26, 0), (27, 1), (28, -1), (29, 1), (30, 1)]), (15, [(1, 1), (2, 1), (3, 0), (4, 1), (5, 0), (6, 0), (7, -1), (8, 1), (9, 0), (10, 0), (11, -1), (12, 0), (13, -1), (14, -1), (15, 0), (16, 1), (17, 1), (18, 0), (19, 1), (20, 0), (21, 0), (22, -1), (23, 1), (24, 0), (25, 0), (26, -1), (27, 0), (28, -1), (29, -1), (30, 0)]), (17, [(1, 1), (2, 1), (3, -1), (4, 1), (5, -1), (6, -1), (7, -1), (8, 1), (9, 1), (10, -1), (11, -1), (12, -1), (13, 1), (14, -1), (15, 1), (16, 1), (17, 0), (18, 1), (19, 1), (20, -1), (21, 1), (22, -1), (23, -1), (24, -1), (25, 1), (26, 1), (27, -1), (28, -1), (29, -1), (30, 1)]), (19, [(1, 1), (2, -1), (3, -1), (4, 1), (5, 1), (6, 1), (7, 1), (8, -1), (9, 1), (10, -1), (11, 1), (12, -1), (13, -1), (14, -1), (15, -1), (16, 1), (17, 1), (18, -1), (19, 0), (20, 1), (21, -1), (22, -1), (23, 1), (24, 1), (25, 1), (26, 1), (27, -1), (28, 1), (29, -1), (30, 1)]), (21, [(1, 1), (2, -1), (3, 0), (4, 1), (5, 1), (6, 0), (7, 0), (8, -1), (9, 0), (10, -1), (11, -1), (12, 0), (13, -1), (14, 0), (15, 0), (16, 1), (17, 1), (18, 0), (19, -1), (20, 1), (21, 0), (22, 1), (23, -1), (24, 0), (25, 1), (26, 1), (27, 0), (28, 0), (29, -1), (30, 0)]), (23, [(1, 1), (2, 1), (3, 1), (4, 1), (5, -1), (6, 1), (7, -1), (8, 1), (9, 1), (10, -1), (11, -1), (12, 1), (13, 1), (14, -1), (15, -1), (16, 1), (17, -1), (18, 1), (19, -1), (20, -1), (21, -1), (22, -1), (23, 0), (24, 1), (25, 1), (26, 1), (27, 1), (28, -1), (29, 1), (30, -1)]), (25, [(1, 1), (2, 1), (3, 1), (4, 1), (5, 0), (6, 1), (7, 1), (8, 1), (9, 1), (10, 0), (11, 1), (12, 1), (13, 1), (14, 1), (15, 0), (16, 1), (17, 1), (18, 1), (19, 1), (20, 0), (21, 1), (22, 1), (23, 1), (24, 1), (25, 0), (26, 1), (27, 1), (28, 1), (29, 1), (30, 0)]), (27, [(1, 1), (2, -1), (3, 0), (4, 1), (5, -1), (6, 0), (7, 1), (8, -1), (9, 0), (10, 1), (11, -1), (12, 0), (13, 1), (14, -1), (15, 0), (16, 1), (17, -1), (18, 0), (19, 1), (20, -1), (21, 0), (22, 1), (23, -1), (24, 0), (25, 1), (26, -1), (27, 0), (28, 1), (29, -1), (30, 0)]), (29, [(1, 1), (2, -1), (3, -1), (4, 1), (5, 1), (6, 1), (7, 1), (8, -1), (9, 1), (10, -1), (11, -1), (12, -1), (13, 1), (14, -1), (15, -1), (16, 1), (17, -1), (18, -1), (19, -1), (20, 1), (21, -1), (22, 1), (23, 1), (24, 1), (25, 1), (26, -1), (27, -1), (28, 1), (29, 0), (30, 1)])]
        js = self.Integer.jacobi_symbol
        for n, kj in tv:
            for k, j in kj:
                self.assertEqual(js(k, n), j)

    def test_hex(self):
        v1, = self.Integers(16)
        self.assertEqual(hex(v1), '0x10')

    def test_mult_modulo_bytes(self):
        modmult = self.Integer._mult_modulo_bytes
        res = modmult(4, 5, 19)
        self.assertEqual(res, b'\x01')
        res = modmult(4 - 19, 5, 19)
        self.assertEqual(res, b'\x01')
        res = modmult(4, 5 - 19, 19)
        self.assertEqual(res, b'\x01')
        res = modmult(4 + 19, 5, 19)
        self.assertEqual(res, b'\x01')
        res = modmult(4, 5 + 19, 19)
        self.assertEqual(res, b'\x01')
        modulus = 2 ** 512 - 1
        t1 = 13 ** 100
        t2 = 17 ** 100
        expect = b"\xfa\xb2\x11\x87\xc3(y\x07\xf8\xf1n\xdepq\x0b\xca\xf3\xd3B,\xef\xf2\xfbf\xcc)\x8dZ*\x95\x98r\x96\xa8\xd5\xc3}\xe2q:\xa2'z\xf48\xde%\xef\t\x07\xbc\xc4[C\x8bUE2\x90\xef\x81\xaa:\x08"
        self.assertEqual(expect, modmult(t1, t2, modulus))
        self.assertRaises(ZeroDivisionError, modmult, 4, 5, 0)
        self.assertRaises(ValueError, modmult, 4, 5, -1)
        self.assertRaises(ValueError, modmult, 4, 5, 4)