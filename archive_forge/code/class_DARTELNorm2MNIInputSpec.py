import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class DARTELNorm2MNIInputSpec(SPMCommandInputSpec):
    template_file = ImageFileSPM(exists=True, copyfile=False, mandatory=True, desc='DARTEL template', field='mni_norm.template')
    flowfield_files = InputMultiPath(ImageFileSPM(exists=True), mandatory=True, desc='DARTEL flow fields u_rc1*', field='mni_norm.data.subjs.flowfields')
    apply_to_files = InputMultiPath(ImageFileSPM(exists=True), desc='Files to apply the transform to', field='mni_norm.data.subjs.images', mandatory=True, copyfile=False)
    voxel_size = traits.Tuple(traits.Float, traits.Float, traits.Float, desc='Voxel sizes for output file', field='mni_norm.vox')
    bounding_box = traits.Tuple(traits.Float, traits.Float, traits.Float, traits.Float, traits.Float, traits.Float, desc='Voxel sizes for output file', field='mni_norm.bb')
    modulate = traits.Bool(field='mni_norm.preserve', desc='Modulate out images - no modulation preserves concentrations')
    fwhm = traits.Either(traits.List(traits.Float(), minlen=3, maxlen=3), traits.Float(), field='mni_norm.fwhm', desc='3-list of fwhm for each dimension')