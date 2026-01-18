from __future__ import annotations
import abc
import typing
from math import gcd
from cryptography.hazmat.primitives import _serialization, hashes
from cryptography.hazmat.primitives._asymmetric import AsymmetricPadding
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
def _modinv(e: int, m: int) -> int:
    """
    Modular Multiplicative Inverse. Returns x such that: (x*e) mod m == 1
    """
    x1, x2 = (1, 0)
    a, b = (e, m)
    while b > 0:
        q, r = divmod(a, b)
        xn = x1 - q * x2
        a, b, x1, x2 = (b, r, x2, xn)
    return x1 % m