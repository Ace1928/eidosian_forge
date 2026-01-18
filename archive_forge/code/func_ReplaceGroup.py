import math
import sys
import time
import numpy
import rdkit.DistanceGeometry as DG
from rdkit import Chem
from rdkit import RDLogger as logging
from rdkit.Chem import ChemicalFeatures, ChemicalForceFields
from rdkit.Chem import rdDistGeom as MolDG
from rdkit.Chem.Pharm3D import ExcludedVolume
from rdkit.ML.Data import Stats
def ReplaceGroup(match, bounds, slop=0.01, useDirs=False, dirLength=defaultFeatLength):
    """ Adds an entry at the end of the bounds matrix for a point at
   the center of a multi-point feature

   returns a 2-tuple:
     new bounds mat
     index of point added

   >>> boundsMat = numpy.array([[0.0, 2.0, 2.0],[1.0, 0.0, 2.0],[1.0, 1.0, 0.0]])
   >>> match = [0, 1, 2]
   >>> bm,idx = ReplaceGroup(match, boundsMat, slop=0.0)

   the index is at the end:

   >>> idx == 3
   True

   and the matrix is one bigger:

   >>> bm.shape == (4, 4)
   True

   but the original bounds mat is not altered:

   >>> boundsMat.shape == (3, 3)
   True


   We make the assumption that the points of the
   feature form a regular polygon, are listed in order
   (i.e. pt 0 is a neighbor to pt 1 and pt N-1)
   and that the replacement point goes at the center:

   >>> print(', '.join([f'{x:.3f}' for x in bm[-1]]))
   0.577, 0.577, 0.577, 0.000
   >>> print(', '.join([f'{x:.3f}' for x in bm[:,-1]]))
   1.155, 1.155, 1.155, 0.000

   The slop argument (default = 0.01) is fractional:

   >>> bm, idx = ReplaceGroup(match, boundsMat)
   >>> print(', '.join([f'{x:.3f}' for x in bm[-1]]))
   0.572, 0.572, 0.572, 0.000
   >>> print(', '.join([f'{x:.3f}' for x in bm[:,-1]]))
   1.166, 1.166, 1.166, 0.000

  """
    maxVal = -1000.0
    minVal = 100000000.0
    nPts = len(match)
    for i in range(nPts):
        idx0 = match[i]
        if i < nPts - 1:
            idx1 = match[i + 1]
        else:
            idx1 = match[0]
        if idx1 < idx0:
            idx0, idx1 = (idx1, idx0)
        minVal = min(minVal, bounds[idx1, idx0])
        maxVal = max(maxVal, bounds[idx0, idx1])
    maxVal *= 1 + slop
    minVal *= 1 - slop
    scaleFact = 1.0 / (2.0 * math.sin(math.pi / nPts))
    minVal *= scaleFact
    maxVal *= scaleFact
    replaceIdx = bounds.shape[0]
    enhanceSize: int = int(bool(useDirs))
    bm = numpy.zeros((bounds.shape[0] + 1 + enhanceSize, bounds.shape[1] + 1 + enhanceSize), dtype=numpy.float64)
    bm[:bounds.shape[0], :bounds.shape[1]] = bounds
    bm[:replaceIdx, replaceIdx] = 1000.0
    if useDirs:
        bm[:replaceIdx + 1, replaceIdx + 1] = 1000.0
        bm[replaceIdx, replaceIdx + 1] = dirLength + slop
        bm[replaceIdx + 1, replaceIdx] = dirLength - slop
    for idx1 in match:
        bm[idx1, replaceIdx] = maxVal
        bm[replaceIdx, idx1] = minVal
        if useDirs:
            bm[idx1, replaceIdx + 1] = numpy.sqrt(bm[replaceIdx, replaceIdx + 1] ** 2 + maxVal ** 2)
            bm[replaceIdx + 1, idx1] = numpy.sqrt(bm[replaceIdx + 1, replaceIdx] ** 2 + minVal ** 2)
    return (bm, replaceIdx)