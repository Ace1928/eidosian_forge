import numpy
from rdkit import Chem, Geometry, RDLogger
from rdkit.Chem.Subshape import SubshapeObjects
from rdkit.Numerics import Alignment
class SubshapeAlignment(object):
    transform = None
    triangleSSD = None
    targetTri = None
    queryTri = None
    alignedConfId = -1
    dirMatch = 0.0
    shapeDist = 0.0