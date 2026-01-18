import copy
from math import *
from qt import *
from qtcanvas import *
from rdkit.sping import pid
def _ColorToQt(color):
    """ convenience function for converting a sping.pid color to a Qt color

  """
    if color == pid.transparent:
        return None
    else:
        return QColor(int(color.red * 255), int(color.green * 255), int(color.blue * 255))