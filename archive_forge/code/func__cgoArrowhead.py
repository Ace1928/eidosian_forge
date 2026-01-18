import math
from rdkit import RDLogger as logging
from rdkit import Geometry
from rdkit.Chem.Features import FeatDirUtilsRD as FeatDirUtils
import os
import sys
from optparse import OptionParser
from rdkit import RDConfig
def _cgoArrowhead(viewer, tail, head, radius, color, label, headFrac=0.3, nSteps=10, aspect=0.5):
    global _globalArrowCGO
    delta = head - tail
    normal = _getVectNormal(delta)
    delta.Normalize()
    dv = head - tail
    dv.Normalize()
    dv *= headFrac
    startP = head
    normal *= headFrac * aspect
    cgo = [BEGIN, TRIANGLE_FAN, COLOR, color[0], color[1], color[2], NORMAL, dv.x, dv.y, dv.z, VERTEX, head.x + dv.x, head.y + dv.y, head.z + dv.z]
    base = [BEGIN, TRIANGLE_FAN, COLOR, color[0], color[1], color[2], NORMAL, -dv.x, -dv.y, -dv.z, VERTEX, head.x, head.y, head.z]
    v = startP + normal
    cgo.extend([NORMAL, normal.x, normal.y, normal.z])
    cgo.extend([VERTEX, v.x, v.y, v.z])
    base.extend([VERTEX, v.x, v.y, v.z])
    for i in range(1, nSteps):
        v = FeatDirUtils.ArbAxisRotation(360.0 / nSteps * i, delta, normal)
        cgo.extend([NORMAL, v.x, v.y, v.z])
        v += startP
        cgo.extend([VERTEX, v.x, v.y, v.z])
        base.extend([VERTEX, v.x, v.y, v.z])
    cgo.extend([NORMAL, normal.x, normal.y, normal.z])
    cgo.extend([VERTEX, startP.x + normal.x, startP.y + normal.y, startP.z + normal.z])
    base.extend([VERTEX, startP.x + normal.x, startP.y + normal.y, startP.z + normal.z])
    cgo.append(END)
    base.append(END)
    cgo.extend(base)
    _globalArrowCGO.extend(cgo)