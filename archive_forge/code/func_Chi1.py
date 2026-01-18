import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def Chi1(mol):
    """ From equations (1),(11) and (12) of Rev. Comp. Chem. vol 2, 367-422, (1991)

  """
    c1s = [x.GetBeginAtom().GetDegree() * x.GetEndAtom().GetDegree() for x in mol.GetBonds()]
    while 0 in c1s:
        c1s.remove(0)
    c1s = numpy.array(c1s, 'd')
    res = sum(numpy.sqrt(1.0 / c1s))
    return res