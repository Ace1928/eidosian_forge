import string
from ..sage_helper import _within_sage, sage_method
class PhiAlpha3:

    def __init__(self, phi, alpha):
        self.base_ring = phi.range()
        self.image_ring = MatrixSpace(self.base_ring, 3)
        self.phi, self.alpha = (phi, alpha)

    def range(self):
        return self.image_ring

    def __call__(self, word):
        a = self.phi(word)
        A = adjoint_action(self.alpha(word))
        M = self.image_ring(0)
        for i in range(3):
            for j in range(3):
                M[i, j] = a * A[i, j]
        return M