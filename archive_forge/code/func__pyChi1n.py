import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def _pyChi1n(mol):
    """  Similar to Hall Kier Chi1v, but uses nVal instead of valence

  """
    delts = numpy.array([_nVal(x) for x in mol.GetAtoms()], 'd')
    res = 0.0
    for bond in mol.GetBonds():
        v = delts[bond.GetBeginAtomIdx()] * delts[bond.GetEndAtomIdx()]
        if v != 0.0:
            res += numpy.sqrt(1.0 / v)
    return res