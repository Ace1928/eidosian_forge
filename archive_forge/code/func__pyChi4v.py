import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def _pyChi4v(mol):
    """ From equations (5),(15) and (16) of Rev. Comp. Chem. vol 2, 367-422, (1991)

  **NOTE**: because the current path finding code does, by design,
  detect rings as paths (e.g. in C1CC1 there is *1* atom path of
  length 3), values of Chi4v may give results that differ from those
  provided by the old code in molecules that have 3 rings.

  """
    return _pyChiNv_(mol, 4)