import math
import numpy
from rdkit import Geometry
from rdkit.Chem.Subshape import SubshapeObjects
def ComputeGridIndices(shapeGrid, winRad):
    if getattr(shapeGrid, '_indicesInSphere', None):
        return shapeGrid._indicesInSphere
    gridSpacing = shapeGrid.GetSpacing()
    dX = shapeGrid.GetNumX()
    dY = shapeGrid.GetNumY()
    radInGrid = int(winRad / gridSpacing)
    indicesInSphere = []
    for i in range(-radInGrid, radInGrid + 1):
        for j in range(-radInGrid, radInGrid + 1):
            for k in range(-radInGrid, radInGrid + 1):
                d = int(math.sqrt(i * i + j * j + k * k))
                if d <= radInGrid:
                    idx = (i * dY + j) * dX + k
                    indicesInSphere.append(idx)
    shapeGrid._indicesInSphere = indicesInSphere
    return indicesInSphere