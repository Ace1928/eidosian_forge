import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
def _vtk_to_coords(self, in_file, out_file=None):
    from ..vtkbase import tvtk
    from ...interfaces import vtkbase as VTKInfo
    if VTKInfo.no_tvtk():
        raise ImportError('TVTK is required and tvtk package was not found')
    reader = tvtk.PolyDataReader(file_name=in_file + '.vtk')
    reader.update()
    mesh = VTKInfo.vtk_output(reader)
    points = mesh.points
    if out_file is None:
        out_file, _ = op.splitext(in_file) + '.txt'
    np.savetxt(out_file, points)
    return out_file