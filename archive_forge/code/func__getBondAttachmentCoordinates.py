import copy
import functools
import math
import numpy
from rdkit import Chem
def _getBondAttachmentCoordinates(self, p1, p2, labelSize):
    newpos = [None, None]
    if labelSize is not None:
        labelSizeOffset = [labelSize[0][0] / 2 + cmp(p2[0], p1[0]) * labelSize[0][2], labelSize[0][1] / 2]
        if p1[1] == p2[1]:
            newpos[0] = p1[0] + cmp(p2[0], p1[0]) * labelSizeOffset[0]
        elif abs(labelSizeOffset[1] * (p2[0] - p1[0]) / (p2[1] - p1[1])) < labelSizeOffset[0]:
            newpos[0] = p1[0] + cmp(p2[0], p1[0]) * abs(labelSizeOffset[1] * (p2[0] - p1[0]) / (p2[1] - p1[1]))
        else:
            newpos[0] = p1[0] + cmp(p2[0], p1[0]) * labelSizeOffset[0]
        if p1[0] == p2[0]:
            newpos[1] = p1[1] + cmp(p2[1], p1[1]) * labelSizeOffset[1]
        elif abs(labelSizeOffset[0] * (p1[1] - p2[1]) / (p2[0] - p1[0])) < labelSizeOffset[1]:
            newpos[1] = p1[1] + cmp(p2[1], p1[1]) * abs(labelSizeOffset[0] * (p1[1] - p2[1]) / (p2[0] - p1[0]))
        else:
            newpos[1] = p1[1] + cmp(p2[1], p1[1]) * labelSizeOffset[1]
    else:
        newpos = copy.deepcopy(p1)
    return newpos