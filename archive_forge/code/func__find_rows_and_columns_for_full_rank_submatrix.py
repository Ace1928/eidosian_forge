from .verificationError import *
from snappy.snap import t3mlite as t3m
from sage.all import vector, matrix, prod, exp, RealDoubleField, sqrt
import sage.all
def _find_rows_and_columns_for_full_rank_submatrix(m, expected_rank):
    num_rows, num_cols = m.dimensions()
    rows_left = set(range(num_rows))
    cols_left = set(range(num_cols))
    for i in range(expected_rank):
        row, col = _find_max(m, rows_left, cols_left)
        rows_left.remove(row)
        cols_left.remove(col)
        for r in rows_left:
            m.add_multiple_of_row(r, row, -m[r, col] / m[row, col])
        for c in cols_left:
            m.add_multiple_of_column(c, col, -m[row, c] / m[row, col])
    return ([row for row in range(num_rows) if row not in rows_left], [col for col in range(num_cols) if col not in cols_left])