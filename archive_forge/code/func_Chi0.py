import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def Chi0(mol):
    """ From equations (1),(9) and (10) of Rev. Comp. Chem. vol 2, 367-422, (1991)

  """
    deltas = [x.GetDegree() for x in mol.GetAtoms()]
    while 0 in deltas:
        deltas.remove(0)
    deltas = numpy.array(deltas, 'd')
    res = sum(numpy.sqrt(1.0 / deltas))
    return res