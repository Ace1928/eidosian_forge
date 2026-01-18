import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def _pyKappa3(mol):
    """  Hall-Kier Kappa3 value

   From equations (58), (61) and (62) of Rev. Comp. Chem. vol 2, 367-422, (1991)

  """
    P3 = len(Chem.FindAllPathsOfLengthN(mol, 3))
    A = mol.GetNumHeavyAtoms()
    alpha = HallKierAlpha(mol)
    denom = (P3 + alpha) ** 2
    if denom:
        if A % 2 == 1:
            kappa = (A + alpha - 1) * (A + alpha - 3) ** 2 / denom
        else:
            kappa = (A + alpha - 2) * (A + alpha - 3) ** 2 / denom
    else:
        kappa = 0
    return kappa