from builtins import range
import os
from glob import glob
from .base import ANTSCommand, ANTSCommandInputSpec
from ..base import TraitedSpec, File, traits, isdefined, OutputMultiPath
from ...utils.filemanip import split_filename
class antsIntroductionOutputSpec(TraitedSpec):
    affine_transformation = File(exists=True, desc='affine (prefix_Affine.txt)')
    warp_field = File(exists=True, desc='warp field (prefix_Warp.nii)')
    inverse_warp_field = File(exists=True, desc='inverse warp field (prefix_InverseWarp.nii)')
    input_file = File(exists=True, desc='input image (prefix_repaired.nii)')
    output_file = File(exists=True, desc='output image (prefix_deformed.nii)')