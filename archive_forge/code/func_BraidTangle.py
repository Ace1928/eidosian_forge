import pickle
from .links import Crossing, Strand, Link
from . import planar_isotopy
def BraidTangle(gens, n=None):
    """
    Create an (n,n) tangle from a braid word.

    Input:

    * gens is a list of nonzero integers, positive for the positive generator
      and negative for the negative generator
    * n is the number of strands. By default it is inferred to be the least
      number of strands that works for the given list of generators

    >>> BraidTangle([], 1)
    <Tangle: IdentityBraid(1)>
    >>> BraidTangle([1]).describe()
    'Tangle[{1,2}, {3,4}, X[2,4,3,1]]'
    >>> BraidTangle([-1]).describe()
    'Tangle[{1,2}, {3,4}, X[1,2,4,3]]'
    >>> BraidTangle([1],3).describe()
    'Tangle[{1,2,3}, {4,5,6}, P[3,6], X[2,5,4,1]]'
    >>> BraidTangle([2],3).describe()
    'Tangle[{1,2,3}, {4,5,6}, P[1,4], X[3,6,5,2]]'
    >>> BraidTangle([1,2]).describe()
    'Tangle[{1,2,3}, {4,5,6}, X[7,5,4,1], X[3,6,7,2]]'
    >>> BraidTangle([1,2,1]).describe()
    'Tangle[{1,2,3}, {4,5,6}, X[7,5,4,8], X[3,6,7,9], X[2,9,8,1]]'
    """
    if n is None:
        n = max(-min(gens), max(gens)) + 1

    def gen(i):
        g = OneTangle() if i < 0 else MinusOneTangle()
        return IdentityBraid(abs(i) - 1) | g | IdentityBraid(n - abs(i) - 1)
    b = IdentityBraid(n)
    for i in gens:
        if i == 0:
            raise ValueError('Generators must be nonzero integers')
        if abs(i) >= n:
            raise ValueError('Generators must have magnitude less than n')
        b = b * gen(i)
    return b