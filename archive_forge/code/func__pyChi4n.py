import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def _pyChi4n(mol):
    """  Similar to Hall Kier Chi4v, but uses nVal instead of valence
  This makes a big difference after we get out of the first row.


  **NOTE**: because the current path finding code does, by design,
  detect rings as paths (e.g. in C1CC1 there is *1* atom path of
  length 3), values of Chi4n may give results that differ from those
  provided by the old code in molecules that have 3 rings.

  """
    return _pyChiNn_(mol, 4)