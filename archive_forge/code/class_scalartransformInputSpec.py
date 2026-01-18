import os
from ...base import (
class scalartransformInputSpec(CommandLineInputSpec):
    input_image = File(desc='Image to transform', exists=True, argstr='--input_image %s')
    output_image = traits.Either(traits.Bool, File(), hash_files=False, desc='The transformed image', argstr='--output_image %s')
    transformation = traits.Either(traits.Bool, File(), hash_files=False, desc='Output file for transformation parameters', argstr='--transformation %s')
    invert = traits.Bool(desc='Invert transform before applying.', argstr='--invert ')
    deformation = File(desc='Deformation field.', exists=True, argstr='--deformation %s')
    h_field = traits.Bool(desc='The deformation is an h-field.', argstr='--h_field ')
    interpolation = traits.Enum('nearestneighbor', 'linear', 'cubic', desc='Interpolation type (nearestneighbor, linear, cubic)', argstr='--interpolation %s')