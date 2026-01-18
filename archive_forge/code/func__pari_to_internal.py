from ..pari import pari
import fractions
def _pari_to_internal(m):
    num_cols = len(m)
    if num_cols == 0:
        return []
    num_rows = len(m[0])

    def convert(p):
        d = int(p.denominator())
        n = int(p.numerator())
        if d == 1:
            return n
        return fractions.Fraction(n, d)
    return [[convert(m[r, c]) for c in range(num_cols)] for r in range(num_rows)]