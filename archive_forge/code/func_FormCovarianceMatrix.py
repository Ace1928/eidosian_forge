import math
import numpy
def FormCovarianceMatrix(mat):
    """ form and return the covariance matrix

  """
    nPts = mat.shape[0]
    sumVect = sum(mat)
    sumVect /= float(nPts)
    for row in mat:
        row -= sumVect
    return numpy.dot(numpy.transpose(mat), mat) / (nPts - 1)