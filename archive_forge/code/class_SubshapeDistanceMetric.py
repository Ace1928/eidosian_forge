import numpy
from rdkit import Chem, Geometry, RDLogger
from rdkit.Chem.Subshape import SubshapeObjects
from rdkit.Numerics import Alignment
class SubshapeDistanceMetric(object):
    TANIMOTO = 0
    PROTRUDE = 1