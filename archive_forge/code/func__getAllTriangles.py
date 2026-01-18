import numpy
from rdkit import Chem, Geometry, RDLogger
from rdkit.Chem.Subshape import SubshapeObjects
from rdkit.Numerics import Alignment
def _getAllTriangles(pts, orderedTraversal=False):
    for i in range(len(pts)):
        if orderedTraversal:
            jStart = i + 1
        else:
            jStart = 0
        for j in range(jStart, len(pts)):
            if j == i:
                continue
            if orderedTraversal:
                kStart = j + 1
            else:
                kStart = 0
            for k in range(j + 1, len(pts)):
                if k == i or k == j:
                    continue
                yield (i, j, k)