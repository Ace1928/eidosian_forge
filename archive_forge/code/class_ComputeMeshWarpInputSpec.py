import os.path as op
import numpy as np
from numpy import linalg as nla
from .. import logging
from ..interfaces.base import (
from ..interfaces.vtkbase import tvtk
from ..interfaces import vtkbase as VTKInfo
class ComputeMeshWarpInputSpec(BaseInterfaceInputSpec):
    surface1 = File(exists=True, mandatory=True, desc='Reference surface (vtk format) to which compute distance.')
    surface2 = File(exists=True, mandatory=True, desc='Test surface (vtk format) from which compute distance.')
    metric = traits.Enum('euclidean', 'sqeuclidean', usedefault=True, desc='norm used to report distance')
    weighting = traits.Enum('none', 'area', usedefault=True, desc='"none": no weighting is performed, surface": edge distance is weighted by the corresponding surface area')
    out_warp = File('surfwarp.vtk', usedefault=True, desc='vtk file based on surface1 and warpings mapping it to surface2')
    out_file = File('distance.npy', usedefault=True, desc='numpy file keeping computed distances and weights')