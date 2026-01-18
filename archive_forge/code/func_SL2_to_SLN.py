import string
from ..sage_helper import _within_sage, sage_method
def SL2_to_SLN(A, N):
    F = A.base_ring()
    R = PolynomialRing(F, ['x', 'y'])
    x, y = R.gens()
    X, Y = A * vector(R, (x, y))
    monomials = [x ** (N - 1 - i) * y ** i for i in range(N)]
    image_vectors = [m(X, Y) for m in monomials]
    return matrix(F, [[v.monomial_coefficient(m) for m in monomials] for v in image_vectors])