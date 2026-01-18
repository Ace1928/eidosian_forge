import os
import numpy
from rdkit import Chem, RDConfig
from rdkit.Chem import rdMolDescriptors
def _pyMolLogP(inMol, patts=None, order=None, verbose=0, addHs=1):
    """ DEPRECATED
  """
    if addHs < 0:
        mol = Chem.AddHs(inMol, 1)
    elif addHs > 0:
        mol = Chem.AddHs(inMol, 0)
    else:
        mol = inMol
    if patts is None:
        global _smartsPatterns, _patternOrder
        if _smartsPatterns == {}:
            _patternOrder, _smartsPatterns = _ReadPatts(defaultPatternFileName)
        patts = _smartsPatterns
        order = _patternOrder
    atomContribs = _pyGetAtomContribs(mol, patts, order, verbose=verbose)
    return numpy.sum(atomContribs, 0)[0]