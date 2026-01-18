from math import *
from rdkit.sping.PDF import pdfmetrics  # for font info
from rdkit.sping.pid import *
def _ColorToSVG(color):
    """ convenience function for converting a sping.pid color to an SVG color

  """
    if color == transparent:
        return 'none'
    else:
        return 'rgb(%d,%d,%d)' % (int(color.red * 255), int(color.green * 255), int(color.blue * 255))