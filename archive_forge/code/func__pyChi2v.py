import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def _pyChi2v(mol):
    """ From equations (5),(15) and (16) of Rev. Comp. Chem. vol 2, 367-422, (1991)

  """
    return _pyChiNv_(mol, 2)