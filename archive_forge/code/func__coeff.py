def _coeff(self):
    """Salzer summation weights (aka, "Stehfest coefficients")
        only depend on the approximation order (M) and the precision"""
    M = self.degree
    M2 = int(M / 2)
    V = self.ctx.matrix(M, 1)
    for k in range(1, M + 1):
        z = self.ctx.matrix(min(k, M2) + 1, 1)
        for j in range(int((k + 1) / 2), min(k, M2) + 1):
            z[j] = self.ctx.power(j, M2) * self.ctx.fac(2 * j) / (self.ctx.fac(M2 - j) * self.ctx.fac(j) * self.ctx.fac(j - 1) * self.ctx.fac(k - j) * self.ctx.fac(2 * j - k))
        V[k - 1] = self.ctx.power(-1, k + M2) * self.ctx.fsum(z)
    return V