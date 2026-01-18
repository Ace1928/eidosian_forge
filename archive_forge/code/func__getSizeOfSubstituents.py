import re
from collections import Counter, defaultdict, namedtuple
import numpy as np
import seaborn as sns
from numpy import linalg
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
def _getSizeOfSubstituents(sub, sharedNeighbors, weighdownShared=True):
    if weighdownShared:
        return sum((1.0 / sharedNeighbors[a] for a in sub))
    else:
        return len(sub)