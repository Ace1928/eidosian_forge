import binascii
import struct
import itertools
from Cryptodome.Util.py3compat import bchr, bord, tobytes, tostr, iter_range
from Cryptodome import Random
from Cryptodome.IO import PKCS8, PEM
from Cryptodome.Hash import SHA256
from Cryptodome.Util.asn1 import (
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Math.Primality import (test_probable_prime, COMPOSITE,
from Cryptodome.PublicKey import (_expand_subject_public_key_info,
importKey = import_key
def _generate_domain(L, randfunc):
    """Generate a new set of DSA domain parameters"""
    N = {1024: 160, 2048: 224, 3072: 256}.get(L)
    if N is None:
        raise ValueError('Invalid modulus length (%d)' % L)
    outlen = SHA256.digest_size * 8
    n = (L + outlen - 1) // outlen - 1
    b_ = L - 1 - n * outlen
    q = Integer(4)
    upper_bit = 1 << N - 1
    while test_probable_prime(q, randfunc) != PROBABLY_PRIME:
        seed = randfunc(64)
        U = Integer.from_bytes(SHA256.new(seed).digest()) & upper_bit - 1
        q = U | upper_bit | 1
    assert q.size_in_bits() == N
    offset = 1
    upper_bit = 1 << L - 1
    while True:
        V = [SHA256.new(seed + Integer(offset + j).to_bytes()).digest() for j in iter_range(n + 1)]
        V = [Integer.from_bytes(v) for v in V]
        W = sum([V[i] * (1 << i * outlen) for i in iter_range(n)], (V[n] & (1 << b_) - 1) * (1 << n * outlen))
        X = Integer(W + upper_bit)
        assert X.size_in_bits() == L
        c = X % (q * 2)
        p = X - (c - 1)
        if p.size_in_bits() == L and test_probable_prime(p, randfunc) == PROBABLY_PRIME:
            break
        offset += n + 1
    e = (p - 1) // q
    for count in itertools.count(1):
        U = seed + b'ggen' + bchr(1) + Integer(count).to_bytes()
        W = Integer.from_bytes(SHA256.new(U).digest())
        g = pow(W, e, p)
        if g != 1:
            break
    return (p, q, g, seed)