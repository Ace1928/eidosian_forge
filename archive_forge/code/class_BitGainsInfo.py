import sys
from rdkit import Chem
from rdkit.Chem.rdfragcatalog import *
class BitGainsInfo(object):
    id = -1
    description = ''
    gain = 0.0
    nPerClass = None