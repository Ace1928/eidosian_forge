import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class NewSegmentOutputSpec(TraitedSpec):
    native_class_images = traits.List(traits.List(File(exists=True)), desc='native space probability maps')
    dartel_input_images = traits.List(traits.List(File(exists=True)), desc='dartel imported class images')
    normalized_class_images = traits.List(traits.List(File(exists=True)), desc='normalized class images')
    modulated_class_images = traits.List(traits.List(File(exists=True)), desc='modulated+normalized class images')
    transformation_mat = OutputMultiPath(File(exists=True), desc='Normalization transformation')
    bias_corrected_images = OutputMultiPath(File(exists=True), desc='bias corrected images')
    bias_field_images = OutputMultiPath(File(exists=True), desc='bias field images')
    forward_deformation_field = OutputMultiPath(File(exists=True))
    inverse_deformation_field = OutputMultiPath(File(exists=True))