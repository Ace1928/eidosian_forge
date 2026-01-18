import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class ApplyWarpInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, argstr='--in=%s', mandatory=True, position=0, desc='image to be warped')
    out_file = File(argstr='--out=%s', genfile=True, position=2, desc='output filename', hash_files=False)
    ref_file = File(exists=True, argstr='--ref=%s', mandatory=True, position=1, desc='reference image')
    field_file = File(exists=True, argstr='--warp=%s', desc='file containing warp field')
    abswarp = traits.Bool(argstr='--abs', xor=['relwarp'], desc="treat warp field as absolute: x' = w(x)")
    relwarp = traits.Bool(argstr='--rel', xor=['abswarp'], position=-1, desc="treat warp field as relative: x' = x + w(x)")
    datatype = traits.Enum('char', 'short', 'int', 'float', 'double', argstr='--datatype=%s', desc='Force output data type [char short int float double].')
    supersample = traits.Bool(argstr='--super', desc='intermediary supersampling of output, default is off')
    superlevel = traits.Either(traits.Enum('a'), traits.Int, argstr='--superlevel=%s', desc="level of intermediary supersampling, a for 'automatic' or integer level. Default = 2")
    premat = File(exists=True, argstr='--premat=%s', desc='filename for pre-transform (affine matrix)')
    postmat = File(exists=True, argstr='--postmat=%s', desc='filename for post-transform (affine matrix)')
    mask_file = File(exists=True, argstr='--mask=%s', desc='filename for mask image (in reference space)')
    interp = traits.Enum('nn', 'trilinear', 'sinc', 'spline', argstr='--interp=%s', position=-2, desc='interpolation method')