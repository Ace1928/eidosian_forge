import random
from bisect import bisect
from itertools import count
from array import array as _array
from sympy.core.function import Function
from sympy.core.singleton import S
from .primetest import isprime
from sympy.utilities.misc import as_int
class primepi(Function):
    """ Represents the prime counting function pi(n) = the number
        of prime numbers less than or equal to n.

        Algorithm Description:

        In sieve method, we remove all multiples of prime p
        except p itself.

        Let phi(i,j) be the number of integers 2 <= k <= i
        which remain after sieving from primes less than
        or equal to j.
        Clearly, pi(n) = phi(n, sqrt(n))

        If j is not a prime,
        phi(i,j) = phi(i, j - 1)

        if j is a prime,
        We remove all numbers(except j) whose
        smallest prime factor is j.

        Let $x= j \\times a$ be such a number, where $2 \\le a \\le i / j$
        Now, after sieving from primes $\\le j - 1$,
        a must remain
        (because x, and hence a has no prime factor $\\le j - 1$)
        Clearly, there are phi(i / j, j - 1) such a
        which remain on sieving from primes $\\le j - 1$

        Now, if a is a prime less than equal to j - 1,
        $x= j \\times a$ has smallest prime factor = a, and
        has already been removed(by sieving from a).
        So, we do not need to remove it again.
        (Note: there will be pi(j - 1) such x)

        Thus, number of x, that will be removed are:
        phi(i / j, j - 1) - phi(j - 1, j - 1)
        (Note that pi(j - 1) = phi(j - 1, j - 1))

        $\\Rightarrow$ phi(i,j) = phi(i, j - 1) - phi(i / j, j - 1) + phi(j - 1, j - 1)

        So,following recursion is used and implemented as dp:

        phi(a, b) = phi(a, b - 1), if b is not a prime
        phi(a, b) = phi(a, b-1)-phi(a / b, b-1) + phi(b-1, b-1), if b is prime

        Clearly a is always of the form floor(n / k),
        which can take at most $2\\sqrt{n}$ values.
        Two arrays arr1,arr2 are maintained
        arr1[i] = phi(i, j),
        arr2[i] = phi(n // i, j)

        Finally the answer is arr2[1]

        Examples
        ========

        >>> from sympy import primepi, prime, prevprime, isprime
        >>> primepi(25)
        9

        So there are 9 primes less than or equal to 25. Is 25 prime?

        >>> isprime(25)
        False

        It is not. So the first prime less than 25 must be the
        9th prime:

        >>> prevprime(25) == prime(9)
        True

        See Also
        ========

        sympy.ntheory.primetest.isprime : Test if n is prime
        primerange : Generate all primes in a given range
        prime : Return the nth prime
    """

    @classmethod
    def eval(cls, n):
        if n is S.Infinity:
            return S.Infinity
        if n is S.NegativeInfinity:
            return S.Zero
        try:
            n = int(n)
        except TypeError:
            if n.is_real == False or n is S.NaN:
                raise ValueError('n must be real')
            return
        if n < 2:
            return S.Zero
        if n <= sieve._list[-1]:
            return S(sieve.search(n)[0])
        lim = int(n ** 0.5)
        lim -= 1
        lim = max(lim, 0)
        while lim * lim <= n:
            lim += 1
        lim -= 1
        arr1 = [0] * (lim + 1)
        arr2 = [0] * (lim + 1)
        for i in range(1, lim + 1):
            arr1[i] = i - 1
            arr2[i] = n // i - 1
        for i in range(2, lim + 1):
            if arr1[i] == arr1[i - 1]:
                continue
            p = arr1[i - 1]
            for j in range(1, min(n // (i * i), lim) + 1):
                st = i * j
                if st <= lim:
                    arr2[j] -= arr2[st] - p
                else:
                    arr2[j] -= arr1[n // st] - p
            lim2 = min(lim, i * i - 1)
            for j in range(lim, lim2, -1):
                arr1[j] -= arr1[j // i] - p
        return S(arr2[1])