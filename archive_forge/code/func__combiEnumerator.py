import sys
import warnings
from collections import namedtuple
import numpy
from rdkit import DataStructs, ForceField, RDConfig, rdBase
from rdkit.Chem import *
from rdkit.Chem.ChemicalFeatures import *
from rdkit.Chem.EnumerateStereoisomers import (EnumerateStereoisomers,
from rdkit.Chem.rdChemReactions import *
from rdkit.Chem.rdDepictor import *
from rdkit.Chem.rdDistGeom import *
from rdkit.Chem.rdFingerprintGenerator import *
from rdkit.Chem.rdForceFieldHelpers import *
from rdkit.Chem.rdMolAlign import *
from rdkit.Chem.rdMolDescriptors import *
from rdkit.Chem.rdMolEnumerator import *
from rdkit.Chem.rdMolTransforms import *
from rdkit.Chem.rdPartialCharges import *
from rdkit.Chem.rdqueries import *
from rdkit.Chem.rdReducedGraphs import *
from rdkit.Chem.rdShapeHelpers import *
from rdkit.Geometry import rdGeometry
from rdkit.RDLogger import logger
def _combiEnumerator(items, depth=0):
    for item in items[depth]:
        if depth + 1 < len(items):
            v = _combiEnumerator(items, depth + 1)
            for entry in v:
                l = [item]
                l.extend(entry)
                yield l
        else:
            yield [item]