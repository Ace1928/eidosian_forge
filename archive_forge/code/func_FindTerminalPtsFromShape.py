import math
import numpy
from rdkit import Geometry
from rdkit.Chem.Subshape import SubshapeObjects
def FindTerminalPtsFromShape(shape, winRad, fraction, maxGridVal=3):
    pts = Geometry.FindGridTerminalPoints(shape.grid, winRad, fraction)
    termPts = [SubshapeObjects.SkeletonPoint(location=x) for x in pts]
    return termPts