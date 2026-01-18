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
def DownsampleBoundsMatrix(bm, indices, maxThresh=4.0):
    """ Removes rows from a bounds matrix that are that are greater 
  than a threshold value away from a set of other points

  Returns the modfied bounds matrix

  The goal of this function is to remove rows from the bounds matrix
  that correspond to atoms (atomic index) that are likely to be quite far from
  the pharmacophore we're interested in. Because the bounds smoothing
  we eventually have to do is N^3, this can be a big win

   >>> boundsMat = numpy.array([[0.0, 3.0, 4.0],[2.0, 0.0, 3.0],[2.0, 2.0, 0.0]])
   >>> bm = DownsampleBoundsMatrix(boundsMat,(0, ), 3.5)
   >>> bm.shape == (2, 2)
   True

   we don't touch the input matrix:

   >>> boundsMat.shape == (3, 3)
   True

   >>> print(', '.join([f'{x:.3f}' for x in bm[0]]))
   0.000, 3.000
   >>> print(', '.join([f'{x:.3f}' for x in bm[1]]))
   2.000, 0.000

   if the threshold is high enough, we don't do anything:

   >>> boundsMat = numpy.array([[0.0, 4.0, 3.0],[2.0, 0.0, 3.0],[2.0, 2.0, 0.0]])
   >>> bm = DownsampleBoundsMatrix(boundsMat, (0, ), 5.0)
   >>> bm.shape == (3, 3)
   True

   If there's a max value that's close enough to *any* of the indices
   we pass in, we'll keep it:

   >>> boundsMat = numpy.array([[0.0, 4.0, 3.0],[2.0, 0.0, 3.0],[2.0, 2.0, 0.0]])
   >>> bm = DownsampleBoundsMatrix(boundsMat, (0, 1), 3.5)
   >>> bm.shape == (3, 3)
   True

   However, the datatype should not be changed or uprank into np.float64 as default behaviour
   >>> boundsMat = numpy.array([[0.0, 4.0, 3.0],[2.0, 0.0, 3.0],[2.0, 2.0, 0.0]], dtype=numpy.float32)
   >>> bm = DownsampleBoundsMatrix(boundsMat,(0, 1), 3.5)
   >>> bm.dtype == numpy.float64
   False
   >>> bm.dtype == numpy.float32 or numpy.issubdtype(bm.dtype, numpy.float32)
   True
   >>> bm.dtype == boundsMat.dtype or numpy.issubdtype(bm.dtype, boundsMat.dtype)
   True

  """
    nPts = bm.shape[0]
    if len(indices) == 0:
        return numpy.zeros(shape=tuple([0] * len(bm.shape)), dtype=bm.dtype)
    indicesSet = list(set(indices))
    maskMatrix = numpy.zeros(nPts, dtype=numpy.uint8)
    maskMatrix[indicesSet] = 1
    for idx in indicesSet:
        maskMatrix[numpy.nonzero(bm[idx, idx + 1:] < maxThresh)[0] + (idx + 1)] = 1
    keep = numpy.nonzero(maskMatrix)[0]
    if keep.shape[0] == nPts:
        return bm.copy()
    return bm[numpy.ix_(keep, keep)]