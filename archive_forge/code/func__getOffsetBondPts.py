import copy
import functools
import math
import numpy
from rdkit import Chem
def _getOffsetBondPts(self, p1, p2, offsetX, offsetY, lenFrac=None):
    lenFrac = lenFrac or self.drawingOptions.dblBondLengthFrac
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    fracP1 = (p1[0] + offsetX, p1[1] + offsetY)
    frac = (1.0 - lenFrac) / 2
    fracP1 = (fracP1[0] + dx * frac, fracP1[1] + dy * frac)
    fracP2 = (fracP1[0] + dx * lenFrac, fracP1[1] + dy * lenFrac)
    return (fracP1, fracP2)