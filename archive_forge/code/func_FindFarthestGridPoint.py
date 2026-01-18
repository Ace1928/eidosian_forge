import math
import numpy
from rdkit import Geometry
from rdkit.Chem.Subshape import SubshapeObjects
def FindFarthestGridPoint(shape, loc, winRad, maxGridVal):
    """ find the grid point with max occupancy that is furthest from a
    given location
  """
    shapeGrid = shape.grid
    shapeVect = shapeGrid.GetOccupancyVect()
    nGridPts = len(shapeVect)
    dMax = -1
    for i in range(nGridPts):
        if shapeVect[i] < maxGridVal:
            continue
        posI = shapeGrid.GetGridPointLoc(i)
        dst = posI.Distance(loc)
        if dst > dMax:
            dMax = dst
            res = posI
    count, centroid = Geometry.ComputeGridCentroid(shapeGrid, res, winRad)
    res = centroid
    return res