from sympy.core.numbers import Float
from sympy.core.singleton import S
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.polynomials import assoc_laguerre
from sympy.functions.special.spherical_harmonics import Ynm
def Psi_nlm(n, l, m, r, phi, theta, Z=1):
    """
    Returns the Hydrogen wave function psi_{nlm}. It's the product of
    the radial wavefunction R_{nl} and the spherical harmonic Y_{l}^{m}.

    Parameters
    ==========

    n : integer
        Principal Quantum Number which is
        an integer with possible values as 1, 2, 3, 4,...
    l : integer
        ``l`` is the Angular Momentum Quantum Number with
        values ranging from 0 to ``n-1``.
    m : integer
        ``m`` is the Magnetic Quantum Number with values
        ranging from ``-l`` to ``l``.
    r :
        radial coordinate
    phi :
        azimuthal angle
    theta :
        polar angle
    Z :
        atomic number (1 for Hydrogen, 2 for Helium, ...)

    Everything is in Hartree atomic units.

    Examples
    ========

    >>> from sympy.physics.hydrogen import Psi_nlm
    >>> from sympy import Symbol
    >>> r=Symbol("r", positive=True)
    >>> phi=Symbol("phi", real=True)
    >>> theta=Symbol("theta", real=True)
    >>> Z=Symbol("Z", positive=True, integer=True, nonzero=True)
    >>> Psi_nlm(1,0,0,r,phi,theta,Z)
    Z**(3/2)*exp(-Z*r)/sqrt(pi)
    >>> Psi_nlm(2,1,1,r,phi,theta,Z)
    -Z**(5/2)*r*exp(I*phi)*exp(-Z*r/2)*sin(theta)/(8*sqrt(pi))

    Integrating the absolute square of a hydrogen wavefunction psi_{nlm}
    over the whole space leads 1.

    The normalization of the hydrogen wavefunctions Psi_nlm is:

    >>> from sympy import integrate, conjugate, pi, oo, sin
    >>> wf=Psi_nlm(2,1,1,r,phi,theta,Z)
    >>> abs_sqrd=wf*conjugate(wf)
    >>> jacobi=r**2*sin(theta)
    >>> integrate(abs_sqrd*jacobi, (r,0,oo), (phi,0,2*pi), (theta,0,pi))
    1
    """
    n, l, m, r, phi, theta, Z = map(S, [n, l, m, r, phi, theta, Z])
    if n.is_integer and n < 1:
        raise ValueError("'n' must be positive integer")
    if l.is_integer and (not n > l):
        raise ValueError("'n' must be greater than 'l'")
    if m.is_integer and (not abs(m) <= l):
        raise ValueError("|'m'| must be less or equal 'l'")
    return R_nl(n, l, r, Z) * Ynm(l, m, theta, phi).expand(func=True)