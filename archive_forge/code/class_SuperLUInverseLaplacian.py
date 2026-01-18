import networkx as nx
class SuperLUInverseLaplacian(InverseLaplacian):

    def init_solver(self, L):
        import scipy as sp
        self.lusolve = sp.sparse.linalg.factorized(self.L1.tocsc())

    def solve_inverse(self, r):
        rhs = np.zeros(self.n, dtype=self.dtype)
        rhs[r] = 1
        return self.lusolve(rhs[1:])

    def solve(self, rhs):
        s = np.zeros(rhs.shape, dtype=self.dtype)
        s[1:] = self.lusolve(rhs[1:])
        return s