import os
from Cryptodome.Util.py3compat import *
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
class DSADomainTest(unittest.TestCase):

    def test_domain1(self):
        """Verify we can generate new keys in a given domain"""
        dsa_key_1 = DSA.generate(1024)
        domain_params = dsa_key_1.domain()
        dsa_key_2 = DSA.generate(1024, domain=domain_params)
        self.assertEqual(dsa_key_1.p, dsa_key_2.p)
        self.assertEqual(dsa_key_1.q, dsa_key_2.q)
        self.assertEqual(dsa_key_1.g, dsa_key_2.g)
        self.assertEqual(dsa_key_1.domain(), dsa_key_2.domain())

    def _get_weak_domain(self):
        from Cryptodome.Math.Numbers import Integer
        from Cryptodome.Math import Primality
        p = Integer(4)
        while p.size_in_bits() != 1024 or Primality.test_probable_prime(p) != Primality.PROBABLY_PRIME:
            q1 = Integer.random(exact_bits=80)
            q2 = Integer.random(exact_bits=80)
            q = q1 * q2
            z = Integer.random(exact_bits=1024 - 160)
            p = z * q + 1
        h = Integer(2)
        g = 1
        while g == 1:
            g = pow(h, z, p)
            h += 1
        return (p, q, g)

    def test_generate_error_weak_domain(self):
        """Verify that domain parameters with composite q are rejected"""
        domain_params = self._get_weak_domain()
        self.assertRaises(ValueError, DSA.generate, 1024, domain=domain_params)

    def test_construct_error_weak_domain(self):
        """Verify that domain parameters with composite q are rejected"""
        from Cryptodome.Math.Numbers import Integer
        p, q, g = self._get_weak_domain()
        y = pow(g, 89, p)
        self.assertRaises(ValueError, DSA.construct, (y, g, p, q))