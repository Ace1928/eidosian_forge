import networkx as nx
class FullInverseLaplacian(InverseLaplacian):

    def init_solver(self, L):
        self.IL = np.zeros(L.shape, dtype=self.dtype)
        self.IL[1:, 1:] = np.linalg.inv(self.L1.todense())

    def solve(self, rhs):
        s = np.zeros(rhs.shape, dtype=self.dtype)
        s = self.IL @ rhs
        return s

    def solve_inverse(self, r):
        return self.IL[r, 1:]