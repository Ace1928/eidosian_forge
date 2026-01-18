from .basis_data import basis_sets, load_basisset
class BasisFunction:
    """Create a basis function object.

    A basis set is composed of a set of basis functions that are typically constructed as a linear
    combination of primitive Gaussian functions. For instance, a basis function in the STO-3G basis
    set is formed as

    .. math::

        \\psi = a_1 G_1 + a_2 G_2 + a_3 G_3,

    where :math:`a` denotes the contraction coefficients and :math:`G` is a Gaussian function
    defined as

    .. math::

        G = x^l y^m z^n e^{-\\alpha r^2}.

    Each Gaussian function is characterized by the angular momentum numbers :math:`(l, m, n)` that
    determine the type of the orbital, the exponent :math:`\\alpha` and the position vector
    :math:`r = (x, y, z)`. These parameters and the contraction coefficients :math:`a` define
    atomic basis functions. Predefined values of the exponents and contraction coefficients for
    each atomic orbital of a given chemical element can be obtained from reference libraries such as
    the Basis Set Exchange `library <https://www.basissetexchange.org>`_.

    The basis function object created by the BasisFunction class stores all the basis set parameters
    including the angular momentum, exponents, positions and coefficients of the Gaussian functions.

    The basis function object can be easily passed to the functions that compute various types of
    integrals over such functions, e.g., overlap integrals, which are essential for Hartree-Fock
    calculations.

    Args:
        l (tuple[int]): angular momentum numbers of the basis function.
        alpha (array(float)): exponents of the primitive Gaussian functions
        coeff (array(float)): coefficients of the contracted Gaussian functions
        r (array(float)): positions of the Gaussian functions
    """

    def __init__(self, l, alpha, coeff, r):
        self.l = l
        self.alpha = alpha
        self.coeff = coeff
        self.r = r
        self.params = [self.alpha, self.coeff, self.r]