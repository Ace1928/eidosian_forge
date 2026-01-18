from math import *
from rdkit.sping.PDF import pdfmetrics  # for font info
from rdkit.sping.pid import *
def _PointListToSVG(points, dupFirst=0):
    """ convenience function for converting a list of points to a string
      suitable for passing to SVG path operations

  """
    outStr = ''
    for i in range(len(points)):
        outStr = outStr + '%.2f,%.2f ' % (points[i][0], points[i][1])
    if dupFirst == 1:
        outStr = outStr + '%.2f,%.2f' % (points[0][0], points[0][1])
    return outStr