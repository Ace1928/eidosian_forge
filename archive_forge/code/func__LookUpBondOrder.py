import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def _LookUpBondOrder(atom1Id, atom2Id, bondDic):
    """
     Used by BertzCT
  """
    if atom1Id < atom2Id:
        theKey = (atom1Id, atom2Id)
    else:
        theKey = (atom2Id, atom1Id)
    tmp = bondDic[theKey]
    if tmp == Chem.BondType.AROMATIC:
        tmp = 1.5
    else:
        tmp = float(tmp)
    return tmp