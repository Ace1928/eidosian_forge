import re
import numpy
def ReadDataFile(fileName, comment='#', depVarCol=0, dataType=float):
    """ read in the data file and return a tuple of two Numeric arrays:
  (independent variables, dependant variables).

  **ARGUMENTS:**

  - fileName: the fileName

  - comment: the comment character for the file

  - depVarcol: the column number containing the dependant variable

  - dataType: the Numeric short-hand for the data type

  RETURNS:

   a tuple of two Numeric arrays:

    (independent variables, dependant variables).

  """
    inFile = ReFile(fileName)
    dataLines = inFile.readlines()
    nPts = len(dataLines)
    if dataType in [float, numpy.float32, numpy.float64]:
        _convfunc = float
    else:
        _convfunc = int
    nIndVars = len(dataLines[0].split()) - 1
    indVarMat = numpy.zeros((nPts, nIndVars), dataType)
    depVarVect = numpy.zeros(nPts, dataType)
    for i in range(nPts):
        splitLine = dataLines[i].split()
        depVarVect[i] = _convfunc(splitLine[depVarCol])
        del splitLine[depVarCol]
        indVarMat[i, :] = map(_convfunc, splitLine)
    return (indVarMat, depVarVect)