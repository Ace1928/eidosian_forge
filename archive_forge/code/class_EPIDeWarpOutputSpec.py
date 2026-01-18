import os
import numpy as np
import nibabel as nb
import warnings
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import traits, TraitedSpec, InputMultiPath, File, isdefined
from .base import FSLCommand, FSLCommandInputSpec, Info
class EPIDeWarpOutputSpec(TraitedSpec):
    unwarped_file = File(desc='unwarped epi file')
    vsm_file = File(desc='voxel shift map')
    exfdw = File(desc='dewarped functional volume example')
    exf_mask = File(desc='Mask from example functional volume')