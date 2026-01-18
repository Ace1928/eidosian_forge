import os
from Cryptodome.Util.py3compat import *
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
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