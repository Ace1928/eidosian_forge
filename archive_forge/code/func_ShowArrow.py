import math
from rdkit import RDLogger as logging
from rdkit import Geometry
from rdkit.Chem.Features import FeatDirUtilsRD as FeatDirUtils
import os
import sys
from optparse import OptionParser
from rdkit import RDConfig
def ShowArrow(viewer, tail, head, radius, color, label, transparency=0, includeArrowhead=True):
    global _globalArrowCGO
    if transparency:
        _globalArrowCGO.extend([ALPHA, 1 - transparency])
    else:
        _globalArrowCGO.extend([ALPHA, 1])
    _globalArrowCGO.extend([CYLINDER, tail.x, tail.y, tail.z, head.x, head.y, head.z, radius * 0.1, color[0], color[1], color[2], color[0], color[1], color[2]])
    if includeArrowhead:
        _cgoArrowhead(viewer, tail, head, radius, color, label)