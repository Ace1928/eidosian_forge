import math
import numpy
from rdkit import Geometry
from rdkit.Chem.Subshape import SubshapeObjects
def ComputeShapeGridCentroid(pt, shapeGrid, winRad):
    count = 0
    centroid = Geometry.Point3D(0, 0, 0)
    idxI = shapeGrid.GetGridPointIndex(pt)
    shapeGridVect = shapeGrid.GetOccupancyVect()
    indicesInSphere = ComputeGridIndices(shapeGrid, winRad)
    nGridPts = len(shapeGridVect)
    for idxJ in indicesInSphere:
        idx = idxI + idxJ
        if idx >= 0 and idx < nGridPts:
            wt = shapeGridVect[idx]
            tPt = shapeGrid.GetGridPointLoc(idx)
            centroid += tPt * wt
            count += wt
    if not count:
        raise ValueError('found no weight in sphere')
    centroid /= count
    return (count, centroid)