import math
import numpy
def FormCorrelationMatrix(mat):
    """ form and return the covariance matrix

  """
    nVars = len(mat[0])
    N = len(mat)
    res = numpy.zeros((nVars, nVars), 'd')
    for i in range(nVars):
        x = mat[:, i]
        sumX = sum(x)
        sumX2 = sum(x * x)
        for j in range(i, nVars):
            y = mat[:, j]
            sumY = sum(y)
            sumY2 = sum(y * y)
            numerator = N * sum(x * y) - sumX * sumY
            denom = numpy.sqrt((N * sumX2 - sumX ** 2) * (N * sumY2 - sumY ** 2))
            if denom != 0.0:
                res[i, j] = numerator / denom
                res[j, i] = numerator / denom
            else:
                res[i, j] = 0
                res[j, i] = 0
    return res