import networkx as nx
class InverseLaplacian:

    def __init__(self, L, width=None, dtype=None):
        global np
        import numpy as np
        n, n = L.shape
        self.dtype = dtype
        self.n = n
        if width is None:
            self.w = self.width(L)
        else:
            self.w = width
        self.C = np.zeros((self.w, n), dtype=dtype)
        self.L1 = L[1:, 1:]
        self.init_solver(L)

    def init_solver(self, L):
        pass

    def solve(self, r):
        raise nx.NetworkXError('Implement solver')

    def solve_inverse(self, r):
        raise nx.NetworkXError('Implement solver')

    def get_rows(self, r1, r2):
        for r in range(r1, r2 + 1):
            self.C[r % self.w, 1:] = self.solve_inverse(r)
        return self.C

    def get_row(self, r):
        self.C[r % self.w, 1:] = self.solve_inverse(r)
        return self.C[r % self.w]

    def width(self, L):
        m = 0
        for i, row in enumerate(L):
            w = 0
            x, y = np.nonzero(row)
            if len(y) > 0:
                v = y - i
                w = v.max() - v.min() + 1
                m = max(w, m)
        return m