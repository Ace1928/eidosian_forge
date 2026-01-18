import math
import numpy
def TransformPoints(tFormMat, pts):
    """ transforms a set of points using tFormMat

    **Arguments**

      - tFormMat: a numpy array

      - pts: a list of numpy arrays (or a 2D array)

    **Returns**

      a list of numpy arrays

  """
    pts = numpy.array(pts)
    nPts = len(pts)
    avgP = sum(pts) / nPts
    pts = pts - avgP
    res = [None] * nPts
    for i in range(nPts):
        res[i] = numpy.dot(tFormMat, pts[i])
    return res