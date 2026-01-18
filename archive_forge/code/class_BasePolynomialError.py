from sympy.utilities import public
@public
class BasePolynomialError(Exception):
    """Base class for polynomial related exceptions. """

    def new(self, *args):
        raise NotImplementedError('abstract base class')