import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def _pyKappa2(mol):
    """  Hall-Kier Kappa2 value

   From equations (58) and (60) of Rev. Comp. Chem. vol 2, 367-422, (1991)

  """
    P2 = len(Chem.FindAllPathsOfLengthN(mol, 2))
    A = mol.GetNumHeavyAtoms()
    alpha = HallKierAlpha(mol)
    denom = (P2 + alpha) ** 2
    if denom:
        kappa = (A + alpha - 1) * (A + alpha - 2) ** 2 / denom
    else:
        kappa = 0
    return kappa