import os
import numpy
from rdkit import Chem, RDConfig
from rdkit.Chem import rdMolDescriptors
def _Init():
    global _smartsPatterns, _patternOrder
    if _smartsPatterns == {}:
        _patternOrder, _smartsPatterns = _ReadPatts(defaultPatternFileName)