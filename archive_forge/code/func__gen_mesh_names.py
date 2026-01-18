import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
def _gen_mesh_names(self, name, structures):
    path, prefix, ext = split_filename(self.inputs.out_file)
    if name == 'vtk_surfaces':
        vtks = list()
        for struct in structures:
            vtk = prefix + '-' + struct + '_first.vtk'
            vtks.append(op.abspath(vtk))
        return vtks
    if name == 'bvars':
        bvars = list()
        for struct in structures:
            bvar = prefix + '-' + struct + '_first.bvars'
            bvars.append(op.abspath(bvar))
        return bvars
    return None