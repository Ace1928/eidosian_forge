import numpy
def FindMinValInList(mat, nObjs, minIdx=None):
    """ finds the minimum value in a metricMatrix and returns it and its indices

    **Arguments**

     - mat: the metric matrix

     - nObjs: the number of objects to be considered

     - minIdx: the index of the minimum value (value, row and column still need
       to be calculated

    **Returns**

      a 3-tuple containing:

        1) the row
        2) the column
        3) the minimum value itself

    **Notes**

      -this probably ain't the speediest thing on earth

  """
    assert len(mat) == nObjs * (nObjs - 1) / 2, 'bad matrix length in FindMinValInList'
    if minIdx is None:
        minIdx = numpy.argmin(mat)
    nSoFar = 0
    col = 0
    while nSoFar <= minIdx:
        col = col + 1
        nSoFar += col
    row = minIdx - nSoFar + col
    return (row, col, mat[minIdx])