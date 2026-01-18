import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def Ipc(mol, avg=False, dMat=None, forceDMat=False):
    """This returns the information content of the coefficients of the characteristic
    polynomial of the adjacency matrix of a hydrogen-suppressed graph of a molecule.

    'avg = True' returns the information content divided by the total population.

    From Eq 6 of D. Bonchev & N. Trinajstic, J. Chem. Phys. vol 67, 4517-4533 (1977)

  """
    if forceDMat or dMat is None:
        if forceDMat:
            dMat = Chem.GetDistanceMatrix(mol, 0)
            mol._adjMat = dMat
        else:
            try:
                dMat = mol._adjMat
            except AttributeError:
                dMat = Chem.GetDistanceMatrix(mol, 0)
                mol._adjMat = dMat
    adjMat = numpy.equal(dMat, 1)
    cPoly = abs(Graphs.CharacteristicPolynomial(mol, adjMat))
    if avg:
        return entropy.InfoEntropy(cPoly)
    else:
        return sum(cPoly) * entropy.InfoEntropy(cPoly)