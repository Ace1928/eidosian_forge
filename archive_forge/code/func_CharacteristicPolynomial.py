import types
import numpy
from rdkit import Chem, DataStructs
def CharacteristicPolynomial(mol, mat=None):
    """ calculates the characteristic polynomial for a molecular graph

      if mat is not passed in, the molecule's Weighted Adjacency Matrix will
      be used.

      The approach used is the Le Verrier-Faddeev-Frame method described
      in _Chemical Graph Theory, 2nd Edition_ by Nenad Trinajstic (CRC Press,
      1992), pg 76.

    """
    nAtoms = mol.GetNumAtoms()
    if mat is None:
        pass
    else:
        A = mat
    I = 1.0 * numpy.identity(nAtoms)
    An = A
    res = numpy.zeros(nAtoms + 1, float)
    res[0] = 1.0
    for n in range(1, nAtoms + 1):
        res[n] = 1.0 / n * numpy.trace(An)
        Bn = An - res[n] * I
        An = numpy.dot(A, Bn)
    res[1:] *= -1
    return res