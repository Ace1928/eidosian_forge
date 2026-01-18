from rdkit.sping.pid import *
import math
import os
def _setColor(self, c):
    """Set the pen color from a piddle color."""
    self._color = (int(c.red * 255), int(c.green * 255), int(c.blue * 255))