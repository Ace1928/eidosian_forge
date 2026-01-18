import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def _pyChi0v(mol):
    """  From equations (5),(9) and (10) of Rev. Comp. Chem. vol 2, 367-422, (1991)

  """
    deltas = _hkDeltas(mol)
    while 0 in deltas:
        deltas.remove(0)
    mol._hkDeltas = None
    res = sum(numpy.sqrt(1.0 / numpy.array(deltas)))
    return res