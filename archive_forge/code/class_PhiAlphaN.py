import string
from ..sage_helper import _within_sage, sage_method
class PhiAlphaN:

    def __init__(self, phi, alpha, N):
        self.base_ring = phi.range()
        self.image_ring = MatrixSpace(self.base_ring, N)
        self.phi, self.alpha, self.N = (phi, alpha, N)

    def range(self):
        return self.image_ring

    def __call__(self, word):
        a = self.phi(word)
        A = SL2_to_SLN(self.alpha(word), self.N)
        M = self.image_ring(0)
        for i in range(self.N):
            for j in range(self.N):
                M[i, j] = a * A[i, j]
        return M