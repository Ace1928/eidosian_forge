import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class VecRegInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, argstr='-i %s', desc='filename for input vector or tensor field', mandatory=True)
    out_file = File(argstr='-o %s', desc='filename for output registered vector or tensor field', genfile=True, hash_files=False)
    ref_vol = File(exists=True, argstr='-r %s', desc='filename for reference (target) volume', mandatory=True)
    affine_mat = File(exists=True, argstr='-t %s', desc='filename for affine transformation matrix')
    warp_field = File(exists=True, argstr='-w %s', desc='filename for 4D warp field for nonlinear registration')
    rotation_mat = File(exists=True, argstr='--rotmat=%s', desc='filename for secondary affine matrix if set, this will be used for the rotation of the vector/tensor field')
    rotation_warp = File(exists=True, argstr='--rotwarp=%s', desc='filename for secondary warp field if set, this will be used for the rotation of the vector/tensor field')
    interpolation = traits.Enum('nearestneighbour', 'trilinear', 'sinc', 'spline', argstr='--interp=%s', desc='interpolation method : nearestneighbour, trilinear (default), sinc or spline')
    mask = File(exists=True, argstr='-m %s', desc='brain mask in input space')
    ref_mask = File(exists=True, argstr='--refmask=%s', desc='brain mask in output space (useful for speed up of nonlinear reg)')