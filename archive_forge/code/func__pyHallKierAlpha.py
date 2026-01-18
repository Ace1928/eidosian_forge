import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def _pyHallKierAlpha(m):
    """ calculate the Hall-Kier alpha value for a molecule

   From equations (58) of Rev. Comp. Chem. vol 2, 367-422, (1991)

  """
    alphaSum = 0.0
    rC = ptable.GetRb0(6)
    for atom in m.GetAtoms():
        atNum = atom.GetAtomicNum()
        if not atNum:
            continue
        symb = atom.GetSymbol()
        alphaV = hallKierAlphas.get(symb, None)
        if alphaV is not None:
            hyb = atom.GetHybridization() - 2
            if hyb < len(alphaV):
                alpha = alphaV[hyb]
                if alpha is None:
                    alpha = alphaV[-1]
            else:
                alpha = alphaV[-1]
        else:
            rA = ptable.GetRb0(atNum)
            alpha = rA / rC - 1
        alphaSum += alpha
    return alphaSum