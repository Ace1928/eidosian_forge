import os.path as op
import numpy as np
from numpy import linalg as nla
from .. import logging
from ..interfaces.base import (
from ..interfaces.vtkbase import tvtk
from ..interfaces import vtkbase as VTKInfo
class MeshWarpMathsOutputSpec(TraitedSpec):
    out_warp = File(exists=True, desc='vtk file with the vertex-wise mapping of surface1 to surface2')
    out_file = File(exists=True, desc='vtk with surface warped')