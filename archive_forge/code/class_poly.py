import unittest
from array import array
import binascii
from .crcmod import mkCrcFun, Crc
from .crcmod import _usingExtension
from .predefined import PredefinedCrc
from .predefined import mkPredefinedCrcFun
from .predefined import _crc_definitions as _predefined_crc_definitions
class poly:
    """Class implementing polynomials over the field of integers mod 2"""

    def __init__(self, p):
        p = int(p)
        if p < 0:
            raise ValueError('invalid polynomial')
        self.p = p

    def __int__(self):
        return self.p

    def __eq__(self, other):
        return self.p == other.p

    def __ne__(self, other):
        return self.p != other.p

    def __cmp__(self, other):
        return cmp(self.p, other.p)

    def __bool__(self):
        return self.p != 0

    def __neg__(self):
        return self

    def __invert__(self):
        n = max(self.deg() + 1, 1)
        x = (1 << n) - 1
        return poly(self.p ^ x)

    def __add__(self, other):
        return poly(self.p ^ other.p)

    def __sub__(self, other):
        return poly(self.p ^ other.p)

    def __mul__(self, other):
        a = self.p
        b = other.p
        if a == 0 or b == 0:
            return poly(0)
        x = 0
        while b:
            if b & 1:
                x = x ^ a
            a = a << 1
            b = b >> 1
        return poly(x)

    def __divmod__(self, other):
        u = self.p
        m = self.deg()
        v = other.p
        n = other.deg()
        if v == 0:
            raise ZeroDivisionError('polynomial division by zero')
        if n == 0:
            return (self, poly(0))
        if m < n:
            return (poly(0), self)
        k = m - n
        a = 1 << m
        v = v << k
        q = 0
        while k > 0:
            if a & u:
                u = u ^ v
                q = q | 1
            q = q << 1
            a = a >> 1
            v = v >> 1
            k -= 1
        if a & u:
            u = u ^ v
            q = q | 1
        return (poly(q), poly(u))

    def __div__(self, other):
        return self.__divmod__(other)[0]

    def __mod__(self, other):
        return self.__divmod__(other)[1]

    def __repr__(self):
        return 'poly(0x%XL)' % self.p

    def __str__(self):
        p = self.p
        if p == 0:
            return '0'
        lst = {0: [], 1: ['1'], 2: ['x'], 3: ['1', 'x']}[p & 3]
        p = p >> 2
        n = 2
        while p:
            if p & 1:
                lst.append('x^%d' % n)
            p = p >> 1
            n += 1
        lst.reverse()
        return '+'.join(lst)

    def deg(self):
        """return the degree of the polynomial"""
        a = self.p
        if a == 0:
            return -1
        n = 0
        while a >= 65536:
            n += 16
            a = a >> 16
        a = int(a)
        while a > 1:
            n += 1
            a = a >> 1
        return n