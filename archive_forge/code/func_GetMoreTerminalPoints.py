import math
import numpy
from rdkit import Geometry
from rdkit.Chem.Subshape import SubshapeObjects
def GetMoreTerminalPoints(shape, pts, winRad, maxGridVal, targetNumber=5):
    """ adds a set of new terminal points using a max-min algorithm
  """
    shapeGrid = shape.grid
    shapeVect = shapeGrid.GetOccupancyVect()
    nGridPts = len(shapeVect)
    while len(pts) < targetNumber:
        maxMin = -1
        for i in range(nGridPts):
            if shapeVect[i] < maxGridVal:
                continue
            minVal = 100000000.0
            posI = shapeGrid.GetGridPointLoc(i)
            for currPt in pts:
                dst = posI.Distance(currPt.location)
                if dst < minVal:
                    minVal = dst
            if minVal > maxMin:
                maxMin = minVal
                bestPt = posI
        count, centroid = Geometry.ComputeGridCentroid(shapeGrid, bestPt, winRad)
        pts.append(SubshapeObjects.SkeletonPoint(location=centroid))