from sympy.utilities import public
@public
class PolynomialDivisionFailed(BasePolynomialError):

    def __init__(self, f, g, domain):
        self.f = f
        self.g = g
        self.domain = domain

    def __str__(self):
        if self.domain.is_EX:
            msg = "You may want to use a different simplification algorithm. Note that in general it's not possible to guarantee to detect zero in this domain."
        elif not self.domain.is_Exact:
            msg = 'Your working precision or tolerance of computations may be set improperly. Adjust those parameters of the coefficient domain and try again.'
        else:
            msg = "Zero detection is guaranteed in this coefficient domain. This may indicate a bug in SymPy or the domain is user defined and doesn't implement zero detection properly."
        return "couldn't reduce degree in a polynomial division algorithm when dividing %s by %s. This can happen when it's not possible to detect zero in the coefficient domain. The domain of computation is %s. %s" % (self.f, self.g, self.domain, msg)