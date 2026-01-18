import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def _GetCountDict(arr):
    """  *Internal Use Only*

  """
    res = {}
    for v in arr:
        res[v] = res.get(v, 0) + 1
    return res