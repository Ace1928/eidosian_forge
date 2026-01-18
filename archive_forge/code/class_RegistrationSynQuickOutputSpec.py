import os
from ...utils.filemanip import ensure_list
from ..base import TraitedSpec, File, Str, traits, InputMultiPath, isdefined
from .base import ANTSCommand, ANTSCommandInputSpec, LOCAL_DEFAULT_NUMBER_OF_THREADS
class RegistrationSynQuickOutputSpec(TraitedSpec):
    warped_image = File(exists=True, desc='Warped image')
    inverse_warped_image = File(exists=True, desc='Inverse warped image')
    out_matrix = File(exists=True, desc='Affine matrix')
    forward_warp_field = File(exists=True, desc='Forward warp field')
    inverse_warp_field = File(exists=True, desc='Inverse warp field')