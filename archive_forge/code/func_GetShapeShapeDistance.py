import numpy
from rdkit import Chem, Geometry, RDLogger
from rdkit.Chem.Subshape import SubshapeObjects
from rdkit.Numerics import Alignment
def GetShapeShapeDistance(s1, s2, distMetric):
    """ returns the distance between two shapes according to the provided metric """
    if distMetric == SubshapeDistanceMetric.PROTRUDE:
        if s1.grid.GetOccupancyVect().GetTotalVal() < s2.grid.GetOccupancyVect().GetTotalVal():
            d = Geometry.ProtrudeDistance(s1.grid, s2.grid)
        else:
            d = Geometry.ProtrudeDistance(s2.grid, s1.grid)
    else:
        d = Geometry.TanimotoDistance(s1.grid, s2.grid)
    return d