from ..pari import pari
import fractions
def _internal_to_pari(m):
    num_rows = len(m)
    if num_rows == 0:
        return pari.matrix(0, 0)
    num_cols = len(m[0])
    return pari.matrix(num_rows, num_cols, [i for row in m for i in row])