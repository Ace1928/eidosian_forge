import os
from ...utils.filemanip import ensure_list
from ..base import TraitedSpec, File, Str, traits, InputMultiPath, isdefined
from .base import ANTSCommand, ANTSCommandInputSpec, LOCAL_DEFAULT_NUMBER_OF_THREADS
class RegistrationOutputSpec(TraitedSpec):
    forward_transforms = traits.List(File(exists=True), desc='List of output transforms for forward registration')
    reverse_forward_transforms = traits.List(File(exists=True), desc='List of output transforms for forward registration reversed for antsApplyTransform')
    reverse_transforms = traits.List(File(exists=True), desc='List of output transforms for reverse registration')
    forward_invert_flags = traits.List(traits.Bool(), desc='List of flags corresponding to the forward transforms')
    reverse_forward_invert_flags = traits.List(traits.Bool(), desc='List of flags corresponding to the forward transforms reversed for antsApplyTransform')
    reverse_invert_flags = traits.List(traits.Bool(), desc='List of flags corresponding to the reverse transforms')
    composite_transform = File(exists=True, desc='Composite transform file')
    inverse_composite_transform = File(desc='Inverse composite transform file')
    warped_image = File(desc='Outputs warped image')
    inverse_warped_image = File(desc='Outputs the inverse of the warped image')
    save_state = File(desc='The saved registration state to be restored')
    metric_value = traits.Float(desc='the final value of metric')
    elapsed_time = traits.Float(desc='the total elapsed time as reported by ANTs')