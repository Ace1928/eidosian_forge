import copy
import pickle
import time
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
from rdkit.Chem.Subshape import BuilderUtils, SubshapeObjects
class SubshapeCombineOperations(object):
    UNION = 0
    SUM = 1
    INTERSECT = 2