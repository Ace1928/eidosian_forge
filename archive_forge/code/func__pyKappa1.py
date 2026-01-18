import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def _pyKappa1(mol):
    """ Hall-Kier Kappa1 value

   From equations (58) and (59) of Rev. Comp. Chem. vol 2, 367-422, (1991)

  """
    P1 = mol.GetNumBonds(1)
    A = mol.GetNumHeavyAtoms()
    alpha = HallKierAlpha(mol)
    denom = P1 + alpha
    if denom:
        kappa = (A + alpha) * (A + alpha - 1) ** 2 / denom ** 2
    else:
        kappa = 0.0
    return kappa