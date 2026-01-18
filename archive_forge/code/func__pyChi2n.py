import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def _pyChi2n(mol):
    """  Similar to Hall Kier Chi2v, but uses nVal instead of valence
  This makes a big difference after we get out of the first row.

  """
    return _pyChiNn_(mol, 2)