import pickle
from .links import Crossing, Strand, Link
from . import planar_isotopy
def IdentityBraid(n):
    """
    The braid with n strands and no crossings.

    >>> IdentityBraid(0).describe()
    'Tangle[{}, {}]'
    >>> IdentityBraid(1).describe()
    'Tangle[{1}, {2}, P[1,2]]'
    >>> IdentityBraid(2).describe()
    'Tangle[{1,2}, {3,4}, P[1,3], P[2,4]]'
    >>> IdentityBraid(-1)
    Traceback (most recent call last):
        ...
    ValueError: Expecting non-negative int
    """
    if n < 0:
        raise ValueError('Expecting non-negative int')
    strands = [Strand() for i in range(n)]
    entry_points = [(s, 0) for s in strands] + [(s, 1) for s in strands]
    return Tangle(n, strands, entry_points, f'IdentityBraid({n})')