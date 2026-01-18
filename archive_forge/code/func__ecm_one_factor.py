from sympy.ntheory import sieve, isprime
from sympy.core.numbers import mod_inverse
from sympy.core.power import integer_log
from sympy.utilities.misc import as_int
import random
def _ecm_one_factor(n, B1=10000, B2=100000, max_curve=200):
    """Returns one factor of n using
    Lenstra's 2 Stage Elliptic curve Factorization
    with Suyama's Parameterization. Here Montgomery
    arithmetic is used for fast computation of addition
    and doubling of points in elliptic curve.

    This ECM method considers elliptic curves in Montgomery
    form (E : b*y**2*z = x**3 + a*x**2*z + x*z**2) and involves
    elliptic curve operations (mod N), where the elements in
    Z are reduced (mod N). Since N is not a prime, E over FF(N)
    is not really an elliptic curve but we can still do point additions
    and doubling as if FF(N) was a field.

    Stage 1 : The basic algorithm involves taking a random point (P) on an
    elliptic curve in FF(N). The compute k*P using Montgomery ladder algorithm.
    Let q be an unknown factor of N. Then the order of the curve E, |E(FF(q))|,
    might be a smooth number that divides k. Then we have k = l * |E(FF(q))|
    for some l. For any point belonging to the curve E, |E(FF(q))|*P = O,
    hence k*P = l*|E(FF(q))|*P. Thus kP.z_cord = 0 (mod q), and the unknownn
    factor of N (q) can be recovered by taking gcd(kP.z_cord, N).

    Stage 2 : This is a continuation of Stage 1 if k*P != O. The idea utilize
    the fact that even if kP != 0, the value of k might miss just one large
    prime divisor of |E(FF(q))|. In this case we only need to compute the
    scalar multiplication by p to get p*k*P = O. Here a second bound B2
    restrict the size of possible values of p.

    Parameters
    ==========

    n : Number to be Factored
    B1 : Stage 1 Bound
    B2 : Stage 2 Bound
    max_curve : Maximum number of curves generated

    References
    ==========

    .. [1]  Carl Pomerance and Richard Crandall "Prime Numbers:
        A Computational Perspective" (2nd Ed.), page 344
    """
    n = as_int(n)
    if B1 % 2 != 0 or B2 % 2 != 0:
        raise ValueError('The Bounds should be an even integer')
    sieve.extend(B2)
    if isprime(n):
        return n
    from sympy.functions.elementary.miscellaneous import sqrt
    from sympy.polys.polytools import gcd
    D = int(sqrt(B2))
    beta = [0] * (D + 1)
    S = [0] * (D + 1)
    k = 1
    for p in sieve.primerange(1, B1 + 1):
        k *= pow(p, integer_log(B1, p)[0])
    for _ in range(max_curve):
        sigma = rgen.randint(6, n - 1)
        u = (sigma * sigma - 5) % n
        v = 4 * sigma % n
        u_3 = pow(u, 3, n)
        try:
            a24 = pow(v - u, 3, n) * (3 * u + v) * mod_inverse(16 * u_3 * v, n) % n
        except ValueError:
            g = gcd(16 * u_3 * v, n)
            if g == n:
                continue
            return g
        Q = Point(u_3, pow(v, 3, n), a24, n)
        Q = Q.mont_ladder(k)
        g = gcd(Q.z_cord, n)
        if g != 1 and g != n:
            return g
        elif g == n:
            continue
        S[1] = Q.double()
        S[2] = S[1].double()
        beta[1] = S[1].x_cord * S[1].z_cord % n
        beta[2] = S[2].x_cord * S[2].z_cord % n
        for d in range(3, D + 1):
            S[d] = S[d - 1].add(S[1], S[d - 2])
            beta[d] = S[d].x_cord * S[d].z_cord % n
        g = 1
        B = B1 - 1
        T = Q.mont_ladder(B - 2 * D)
        R = Q.mont_ladder(B)
        for r in range(B, B2, 2 * D):
            alpha = R.x_cord * R.z_cord % n
            for q in sieve.primerange(r + 2, r + 2 * D + 1):
                delta = (q - r) // 2
                f = (R.x_cord - S[delta].x_cord) * (R.z_cord + S[delta].z_cord) - alpha + beta[delta]
                g = g * f % n
            T, R = (R, R.add(S[D], T))
        g = gcd(n, g)
        if g != 1 and g != n:
            return g
    raise ValueError('Increase the bounds')