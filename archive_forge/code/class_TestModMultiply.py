import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Util._raw_api import (create_string_buffer,
from Cryptodome.Math._IntegerCustom import _raw_montgomery
class TestModMultiply(unittest.TestCase):

    def test_small(self):
        self.assertEqual(b'\x01', monty_mult(5, 6, 29))

    def test_large(self):
        numbers_len = (modulus1.bit_length() + 7) // 8
        t1 = modulus1 // 2
        t2 = modulus1 - 90
        expect = b'\x00' * (numbers_len - 1) + b'-'
        self.assertEqual(expect, monty_mult(t1, t2, modulus1))

    def test_zero_term(self):
        numbers_len = (modulus1.bit_length() + 7) // 8
        expect = b'\x00' * numbers_len
        self.assertEqual(expect, monty_mult(256, 0, modulus1))
        self.assertEqual(expect, monty_mult(0, 256, modulus1))

    def test_larger_term(self):
        t1 = 2 ** 2047
        expect_int = 18035928840773730123461726118653143896643893030991690902036748754755020857399710871490720025757087193548145641036050602121608319556241438541641764685719326282139487229709863096408212858339365417661850565411065586671059236086983584919881877793677764471633250707743765529980822777591203768782846291836880884647554942482756331246010919919597255670476734478404547223841720108233582957583359908202432652670278551543263176473959823516479696532139257186202613681956701667831367583503063327157476733926912952576973151976430128924002852040335925459672695363457199484737349824310444033539374788015877344967051180584564478675549
        res = bytes_to_long(monty_mult(t1, t1, modulus1))
        self.assertEqual(res, expect_int)