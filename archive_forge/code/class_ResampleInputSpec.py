import os.path as op
import nibabel as nb
import numpy as np
from looseversion import LooseVersion
from ... import logging
from ..base import traits, TraitedSpec, File, isdefined
from .base import (
class ResampleInputSpec(TraitedSpec):
    in_file = File(exists=True, mandatory=True, desc='The input 4D diffusion-weighted image file')
    vox_size = traits.Tuple(traits.Float, traits.Float, traits.Float, desc='specify the new voxel zooms. If no vox_size is set, then isotropic regridding will be performed, with spacing equal to the smallest current zoom.')
    interp = traits.Int(1, mandatory=True, usedefault=True, desc='order of the interpolator (0 = nearest, 1 = linear, etc.')