from sympy.core.numbers import igcd
from sympy.core.power import integer_nthroot
from sympy.core.sympify import sympify
from sympy.external.gmpy import HAS_GMPY
from sympy.utilities.misc import as_int
from mpmath.libmp import bitcount as _bitlength
def _lucas_sequence(n, P, Q, k):
    """Return the modular Lucas sequence (U_k, V_k, Q_k).

    Given a Lucas sequence defined by P, Q, returns the kth values for
    U and V, along with Q^k, all modulo n.  This is intended for use with
    possibly very large values of n and k, where the combinatorial functions
    would be completely unusable.

    The modular Lucas sequences are used in numerous places in number theory,
    especially in the Lucas compositeness tests and the various n + 1 proofs.

    Examples
    ========

    >>> from sympy.ntheory.primetest import _lucas_sequence
    >>> N = 10**2000 + 4561
    >>> sol = U, V, Qk = _lucas_sequence(N, 3, 1, N//2); sol
    (0, 2, 1)

    """
    D = P * P - 4 * Q
    if n < 2:
        raise ValueError('n must be >= 2')
    if k < 0:
        raise ValueError('k must be >= 0')
    if D == 0:
        raise ValueError('D must not be zero')
    if k == 0:
        return _int_tuple(0, 2, Q)
    U = 1
    V = P
    Qk = Q
    b = _bitlength(k)
    if Q == 1:
        while b > 1:
            U = U * V % n
            V = (V * V - 2) % n
            b -= 1
            if k >> b - 1 & 1:
                U, V = (U * P + V, V * P + U * D)
                if U & 1:
                    U += n
                if V & 1:
                    V += n
                U, V = (U >> 1, V >> 1)
    elif P == 1 and Q == -1:
        while b > 1:
            U = U * V % n
            if Qk == 1:
                V = (V * V - 2) % n
            else:
                V = (V * V + 2) % n
                Qk = 1
            b -= 1
            if k >> b - 1 & 1:
                U, V = (U + V, V + U * D)
                if U & 1:
                    U += n
                if V & 1:
                    V += n
                U, V = (U >> 1, V >> 1)
                Qk = -1
    else:
        while b > 1:
            U = U * V % n
            V = (V * V - 2 * Qk) % n
            Qk *= Qk
            b -= 1
            if k >> b - 1 & 1:
                U, V = (U * P + V, V * P + U * D)
                if U & 1:
                    U += n
                if V & 1:
                    V += n
                U, V = (U >> 1, V >> 1)
                Qk *= Q
            Qk %= n
    return _int_tuple(U % n, V % n, Qk)