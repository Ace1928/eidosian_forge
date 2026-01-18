import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _power_exact(self, other, p):
    """Attempt to compute self**other exactly.

        Given Decimals self and other and an integer p, attempt to
        compute an exact result for the power self**other, with p
        digits of precision.  Return None if self**other is not
        exactly representable in p digits.

        Assumes that elimination of special cases has already been
        performed: self and other must both be nonspecial; self must
        be positive and not numerically equal to 1; other must be
        nonzero.  For efficiency, other._exp should not be too large,
        so that 10**abs(other._exp) is a feasible calculation."""
    x = _WorkRep(self)
    xc, xe = (x.int, x.exp)
    while xc % 10 == 0:
        xc //= 10
        xe += 1
    y = _WorkRep(other)
    yc, ye = (y.int, y.exp)
    while yc % 10 == 0:
        yc //= 10
        ye += 1
    if xc == 1:
        xe *= yc
        while xe % 10 == 0:
            xe //= 10
            ye += 1
        if ye < 0:
            return None
        exponent = xe * 10 ** ye
        if y.sign == 1:
            exponent = -exponent
        if other._isinteger() and other._sign == 0:
            ideal_exponent = self._exp * int(other)
            zeros = min(exponent - ideal_exponent, p - 1)
        else:
            zeros = 0
        return _dec_from_triple(0, '1' + '0' * zeros, exponent - zeros)
    if y.sign == 1:
        last_digit = xc % 10
        if last_digit in (2, 4, 6, 8):
            if xc & -xc != xc:
                return None
            e = _nbits(xc) - 1
            emax = p * 93 // 65
            if ye >= len(str(emax)):
                return None
            e = _decimal_lshift_exact(e * yc, ye)
            xe = _decimal_lshift_exact(xe * yc, ye)
            if e is None or xe is None:
                return None
            if e > emax:
                return None
            xc = 5 ** e
        elif last_digit == 5:
            e = _nbits(xc) * 28 // 65
            xc, remainder = divmod(5 ** e, xc)
            if remainder:
                return None
            while xc % 5 == 0:
                xc //= 5
                e -= 1
            emax = p * 10 // 3
            if ye >= len(str(emax)):
                return None
            e = _decimal_lshift_exact(e * yc, ye)
            xe = _decimal_lshift_exact(xe * yc, ye)
            if e is None or xe is None:
                return None
            if e > emax:
                return None
            xc = 2 ** e
        else:
            return None
        if xc >= 10 ** p:
            return None
        xe = -e - xe
        return _dec_from_triple(0, str(xc), xe)
    if ye >= 0:
        m, n = (yc * 10 ** ye, 1)
    else:
        if xe != 0 and len(str(abs(yc * xe))) <= -ye:
            return None
        xc_bits = _nbits(xc)
        if len(str(abs(yc) * xc_bits)) <= -ye:
            return None
        m, n = (yc, 10 ** (-ye))
        while m % 2 == n % 2 == 0:
            m //= 2
            n //= 2
        while m % 5 == n % 5 == 0:
            m //= 5
            n //= 5
    if n > 1:
        if xc_bits <= n:
            return None
        xe, rem = divmod(xe, n)
        if rem != 0:
            return None
        a = 1 << -(-_nbits(xc) // n)
        while True:
            q, r = divmod(xc, a ** (n - 1))
            if a <= q:
                break
            else:
                a = (a * (n - 1) + q) // n
        if not (a == q and r == 0):
            return None
        xc = a
    if xc > 1 and m > p * 100 // _log10_lb(xc):
        return None
    xc = xc ** m
    xe *= m
    if xc > 10 ** p:
        return None
    str_xc = str(xc)
    if other._isinteger() and other._sign == 0:
        ideal_exponent = self._exp * int(other)
        zeros = min(xe - ideal_exponent, p - len(str_xc))
    else:
        zeros = 0
    return _dec_from_triple(0, str_xc + '0' * zeros, xe - zeros)