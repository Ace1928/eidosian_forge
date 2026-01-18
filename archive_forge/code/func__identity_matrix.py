from sympy.polys.monomials import monomial_mul, monomial_div
def _identity_matrix(n, domain):
    M = [[domain.zero] * n for _ in range(n)]
    for i in range(n):
        M[i][i] = domain.one
    return M