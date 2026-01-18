import math
import numpy
def StandardizeMatrix(mat):
    """

  This is the standard *subtract off the average and divide by the deviation*
  standardization function.

   **Arguments**

     - mat: a numpy array

   **Notes**

     - in addition to being returned, _mat_ is modified in place, so **beware**

  """
    nObjs = len(mat)
    avgs = sum(mat, 0) / float(nObjs)
    mat -= avgs
    devs = math.sqrt(sum(mat * mat, 0) / float(nObjs - 1))
    try:
        newMat = mat / devs
    except OverflowError:
        newMat = numpy.zeros(mat.shape, 'd')
        for i in range(mat.shape[1]):
            if devs[i] != 0.0:
                newMat[:, i] = mat[:, i] / devs[i]
    return newMat