import math
import numpy
from rdkit import Geometry
from rdkit.Chem.Subshape import SubshapeObjects
def ExpandTerminalPts(shape, pts, winRad, maxGridVal=3.0, targetNumPts=5):
    """ find additional terminal points until a target number is reached
  """
    if len(pts) == 1:
        pt2 = FindFarthestGridPoint(shape, pts[0].location, winRad, maxGridVal)
        pts.append(SubshapeObjects.SkeletonPoint(location=pt2))
    if len(pts) == 2:
        shapeGrid = shape.grid
        pt1 = pts[0].location
        pt2 = pts[1].location
        center = FindGridPointBetweenPoints(pt1, pt2, shapeGrid, winRad)
        pts.append(SubshapeObjects.SkeletonPoint(location=center))
    if len(pts) < targetNumPts:
        GetMoreTerminalPoints(shape, pts, winRad, maxGridVal, targetNumPts)