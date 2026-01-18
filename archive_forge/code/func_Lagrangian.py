from sympy.utilities import dict_merge
from sympy.utilities.iterables import iterable
from sympy.physics.vector import (Dyadic, Vector, ReferenceFrame,
from sympy.physics.vector.printing import (vprint, vsprint, vpprint, vlatex,
from sympy.physics.mechanics.particle import Particle
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.simplify.simplify import simplify
from sympy.core.backend import (Matrix, sympify, Mul, Derivative, sin, cos,
def Lagrangian(frame, *body):
    """Lagrangian of a multibody system.

    Explanation
    ===========

    This function returns the Lagrangian of a system of Particle's and/or
    RigidBody's. The Lagrangian of such a system is equal to the difference
    between the kinetic energies and potential energies of its constituents. If
    T and V are the kinetic and potential energies of a system then it's
    Lagrangian, L, is defined as

    L = T - V

    The Lagrangian is a scalar.

    Parameters
    ==========

    frame : ReferenceFrame
        The frame in which the velocity or angular velocity of the body is
        defined to determine the kinetic energy.

    body1, body2, body3... : Particle and/or RigidBody
        The body (or bodies) whose Lagrangian is required.

    Examples
    ========

    >>> from sympy.physics.mechanics import Point, Particle, ReferenceFrame
    >>> from sympy.physics.mechanics import RigidBody, outer, Lagrangian
    >>> from sympy import symbols
    >>> M, m, g, h = symbols('M m g h')
    >>> N = ReferenceFrame('N')
    >>> O = Point('O')
    >>> O.set_vel(N, 0 * N.x)
    >>> P = O.locatenew('P', 1 * N.x)
    >>> P.set_vel(N, 10 * N.x)
    >>> Pa = Particle('Pa', P, 1)
    >>> Ac = O.locatenew('Ac', 2 * N.y)
    >>> Ac.set_vel(N, 5 * N.y)
    >>> a = ReferenceFrame('a')
    >>> a.set_ang_vel(N, 10 * N.z)
    >>> I = outer(N.z, N.z)
    >>> A = RigidBody('A', Ac, a, 20, (I, Ac))
    >>> Pa.potential_energy = m * g * h
    >>> A.potential_energy = M * g * h
    >>> Lagrangian(N, Pa, A)
    -M*g*h - g*h*m + 350

    """
    if not isinstance(frame, ReferenceFrame):
        raise TypeError('Please supply a valid ReferenceFrame')
    for e in body:
        if not isinstance(e, (RigidBody, Particle)):
            raise TypeError('*body must have only Particle or RigidBody')
    return kinetic_energy(frame, *body) - potential_energy(*body)