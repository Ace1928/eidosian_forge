import string
from ..sage_helper import _within_sage, sage_method
class MapToAbelianization(SageObject):
    """
    sage: M = Manifold('v2037')
    sage: ab = MapToAbelianization(M.fundamental_group())
    sage: ab.range()
    Multiplicative Abelian group isomorphic to C2 x C4 x Z
    sage: ab('ab').order()
    4
    sage: ab('abc').order()
    +Infinity

    sage: U = Manifold('dLQbcbchecv')
    sage: ab = MapToAbelianization(U.fundamental_group())
    sage: ab.range()
    Multiplicative Abelian group isomorphic to Z
    sage: ab('aaa')
    t^3
    """

    def __init__(self, fund_group):
        self.domain_gens = fund_group.generators()
        ab_words = [abelianize_word(R, self.domain_gens) for R in fund_group.relators()]
        if not ab_words:
            n = fund_group.num_generators()
            self.elementary_divisors = n * [0]
            self.U = identity_matrix(n)
        else:
            R = matrix(ZZ, ab_words).transpose()
            D, U, V = R.smith_form()
            m = U.nrows()
            assert m == D.nrows()
            d = min(D.nrows(), D.ncols())
            diag = D.diagonal()
            num_ones = diag.count(1)
            self.elementary_divisors = diag[num_ones:] + [0] * (m - d)
            self.U = U[num_ones:]
        tor = [d for d in self.elementary_divisors if d != 0]
        free = [d for d in self.elementary_divisors if d == 0]
        names = []
        if len(tor) == 1:
            names.append('u')
        else:
            names += ['u%d' % i for i in range(len(tor))]
        if len(free) == 1:
            names.append('t')
        else:
            names += ['t%d' % i for i in range(len(free))]
        self._range = AbelianGroup(self.elementary_divisors, names=names)

    def range(self):
        return self._range

    def _normalize_exponents(self, exponents):
        D = self.elementary_divisors
        return [v % d if d > 0 else v for v, d in zip(exponents, D)]

    def _exponents_of_word(self, word):
        exponents = self.U * abelianize_word(word, self.domain_gens)
        return self._normalize_exponents(exponents)

    def __call__(self, word):
        return self._range(self._exponents_of_word(word))