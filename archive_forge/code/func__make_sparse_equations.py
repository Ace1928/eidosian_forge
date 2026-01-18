from snappy import snap
from snappy.sage_helper import _within_sage, sage_method
def _make_sparse_equations(self):
    num_eqns = len(self.equations)
    self.sparse_equations = []
    for c in range(num_eqns):
        column = []
        for r in range(num_eqns):
            A, B, dummy = self.equations[r]
            a = A[c]
            b = B[c]
            if a != 0 or b != 0:
                column.append((r, (a, b)))
        self.sparse_equations.append(column)