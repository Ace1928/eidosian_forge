import itertools as it
import numpy as np
from scipy.special import factorial2 as fac2
import pennylane as qml
def _hermite_coulomb(t, u, v, n, p, dr):
    """Evaluate the Hermite integral needed to compute the nuclear attraction and electron repulsion
    integrals.

    These integrals are computed recursively starting from the Boys function
    [`Helgaker (1995) p817 <https://www.worldscientific.com/doi/abs/10.1142/9789812832115_0001>`_]:


    .. math::

        R_{000}^n = (-2p)^n F_n(pR_{CP}^2),

    where :math:`F_n` is the Boys function, :math:`p` is computed from the exponents of the two
    Gaussian functions as :math:`p = \x07lpha + \x08eta`, and :math:`R_{CP}` is the distance between the
    center of the composite Gaussian centered at :math:`P` and the electrostatic potential generated
    by a nucleus at :math:`C`. The following recursive equations are used to compute the
    higher-order Hermite integrals

    .. math::

        R_{t+1, u, v}^n = t R_{t-1, u, v}^{n+1} + x R_{t, u, v}^{n+1},

        R_{t, u+1, v}^n = u R_{t, u-1, v}^{n+1} + y R_{t, u, v}^{n+1},

        R_{t, u, v+1}^n = v R_{t, u, v-1}^{n+1} + z R_{t, u, v}^{n+1},

    where :math:`x`, :math:`y` and :math:`z` are the Cartesian components of :math:`R_{CP}`.

    Args:
        t (integer): order of Hermite derivative in x
        u (integer): order of Hermite derivative in y
        v (float): order of Hermite derivative in z
        n (integer): order of the Boys function
        p (float): sum of the Gaussian exponents
        dr (array[float]): distance between the center of the composite Gaussian and the nucleus

    Returns:
        array[float]: value of the Hermite integral
    """
    x, y, z = dr[0:3]
    T = p * (dr ** 2).sum(axis=0)
    r = 0
    if t == u == v == 0:
        return (-2 * p) ** n * _boys(n, T)
    if t == u == 0:
        if v > 1:
            r = r + (v - 1) * _hermite_coulomb(t, u, v - 2, n + 1, p, dr)
        r = r + z * _hermite_coulomb(t, u, v - 1, n + 1, p, dr)
        return r
    if t == 0:
        if u > 1:
            r = r + (u - 1) * _hermite_coulomb(t, u - 2, v, n + 1, p, dr)
        r = r + y * _hermite_coulomb(t, u - 1, v, n + 1, p, dr)
        return r
    if t > 1:
        r = r + (t - 1) * _hermite_coulomb(t - 2, u, v, n + 1, p, dr)
    r = r + x * _hermite_coulomb(t - 1, u, v, n + 1, p, dr)
    return r