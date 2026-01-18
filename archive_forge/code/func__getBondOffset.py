import copy
import functools
import math
import numpy
from rdkit import Chem
def _getBondOffset(self, p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    ang = math.atan2(dy, dx)
    perp = ang + math.pi / 2.0
    offsetX = math.cos(perp) * self.drawingOptions.dblBondOffset * self.currDotsPerAngstrom
    offsetY = math.sin(perp) * self.drawingOptions.dblBondOffset * self.currDotsPerAngstrom
    return (perp, offsetX, offsetY)