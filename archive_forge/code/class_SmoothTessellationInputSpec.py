import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class SmoothTessellationInputSpec(FSTraitedSpec):
    in_file = File(exists=True, mandatory=True, argstr='%s', position=-2, copyfile=True, desc='Input volume to tessellate voxels from.')
    curvature_averaging_iterations = traits.Int(argstr='-a %d', desc='Number of curvature averaging iterations (default=10)')
    smoothing_iterations = traits.Int(argstr='-n %d', desc='Number of smoothing iterations (default=10)')
    snapshot_writing_iterations = traits.Int(argstr='-w %d', desc='Write snapshot every *n* iterations')
    use_gaussian_curvature_smoothing = traits.Bool(argstr='-g', desc='Use Gaussian curvature smoothing')
    gaussian_curvature_norm_steps = traits.Int(argstr='%d', desc='Use Gaussian curvature smoothing')
    gaussian_curvature_smoothing_steps = traits.Int(argstr=' %d', desc='Use Gaussian curvature smoothing')
    disable_estimates = traits.Bool(argstr='-nw', desc='Disables the writing of curvature and area estimates')
    normalize_area = traits.Bool(argstr='-area', desc='Normalizes the area after smoothing')
    use_momentum = traits.Bool(argstr='-m', desc='Uses momentum')
    out_file = File(argstr='%s', position=-1, genfile=True, desc='output filename or True to generate one')
    out_curvature_file = File(argstr='-c %s', desc='Write curvature to ``?h.curvname`` (default "curv")')
    out_area_file = File(argstr='-b %s', desc='Write area to ``?h.areaname`` (default "area")')
    seed = traits.Int(argstr='-seed %d', desc='Seed for setting random number generator')