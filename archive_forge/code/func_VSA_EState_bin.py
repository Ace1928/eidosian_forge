import bisect
import numpy
from rdkit.Chem.EState.EState import EStateIndices as EStateIndices_
from rdkit.Chem.MolSurf import _LabuteHelper as VSAContribs_
def VSA_EState_bin(mol):
    return VSA_EState_(mol, force=False)[nbin]