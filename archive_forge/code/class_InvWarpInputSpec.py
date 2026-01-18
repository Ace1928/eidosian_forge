import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class InvWarpInputSpec(FSLCommandInputSpec):
    warp = File(exists=True, argstr='--warp=%s', mandatory=True, desc='Name of file containing warp-coefficients/fields. This would typically be the output from the --cout switch of fnirt (but can also use fields, like the output from --fout).')
    reference = File(exists=True, argstr='--ref=%s', mandatory=True, desc='Name of a file in target space. Note that the target space is now different from the target space that was used to create the --warp file. It would typically be the file that was specified with the --in argument when running fnirt.')
    inverse_warp = File(argstr='--out=%s', name_source=['warp'], hash_files=False, name_template='%s_inverse', desc='Name of output file, containing warps that are the "reverse" of those in --warp. This will be a field-file (rather than a file of spline coefficients), and it will have any affine component included as part of the displacements.')
    absolute = traits.Bool(argstr='--abs', xor=['relative'], desc='If set it indicates that the warps in --warp should be interpreted as absolute, provided that it is not created by fnirt (which always uses relative warps). If set it also indicates that the output --out should be absolute.')
    relative = traits.Bool(argstr='--rel', xor=['absolute'], desc='If set it indicates that the warps in --warp should be interpreted as relative. I.e. the values in --warp are displacements from the coordinates in the --ref space. If set it also indicates that the output --out should be relative.')
    niter = traits.Int(argstr='--niter=%d', desc='Determines how many iterations of the gradient-descent search that should be run.')
    regularise = traits.Float(argstr='--regularise=%f', desc='Regularization strength (default=1.0).')
    noconstraint = traits.Bool(argstr='--noconstraint', desc='Do not apply Jacobian constraint')
    jacobian_min = traits.Float(argstr='--jmin=%f', desc='Minimum acceptable Jacobian value for constraint (default 0.01)')
    jacobian_max = traits.Float(argstr='--jmax=%f', desc='Maximum acceptable Jacobian value for constraint (default 100.0)')