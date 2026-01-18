import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def _pyChiNn_(mol, order=2):
    """  Similar to Hall Kier ChiNv, but uses nVal instead of valence
  This makes a big difference after we get out of the first row.

  **NOTE**: because the current path finding code does, by design,
  detect rings as paths (e.g. in C1CC1 there is *1* atom path of
  length 3), values of ChiNn with N >= 3 may give results that differ
  from those provided by the old code in molecules that have rings of
  size 3.

  """
    nval = [_nVal(x) for x in mol.GetAtoms()]
    deltas = numpy.array([1.0 / numpy.sqrt(x) if x else 0.0 for x in nval])
    accum = 0.0
    for path in Chem.FindAllPathsOfLengthN(mol, order + 1, useBonds=0):
        accum += numpy.prod(deltas[numpy.array(path)])
    return accum