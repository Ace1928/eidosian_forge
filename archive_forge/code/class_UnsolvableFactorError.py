from sympy.utilities import public
@public
class UnsolvableFactorError(BasePolynomialError):
    """Raised if ``roots`` is called with strict=True and a polynomial
     having a factor whose solutions are not expressible in radicals
     is encountered."""