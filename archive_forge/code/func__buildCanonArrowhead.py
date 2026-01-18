import math
from rdkit import RDLogger as logging
from rdkit import Geometry
from rdkit.Chem.Features import FeatDirUtilsRD as FeatDirUtils
import os
import sys
from optparse import OptionParser
from rdkit import RDConfig
def _buildCanonArrowhead(headFrac, nSteps, aspect):
    global _canonArrowhead
    startP = RDGeometry.Point3D(0, 0, headFrac)
    _canonArrowhead = [startP]
    scale = headFrac * aspect
    baseV = RDGeometry.Point3D(scale, 0, 0)
    _canonArrowhead.append(baseV)
    twopi = 2 * math.pi
    for i in range(1, nSteps):
        v = RDGeometry.Point3D(scale * math.cos(i * twopi), scale * math.sin(i * twopi), 0)
        _canonArrowhead.append(v)