import os.path as op
from ... import logging
from ..base import (
class EditTransformInputSpec(BaseInterfaceInputSpec):
    transform_file = File(exists=True, mandatory=True, desc='transform-parameter file, only 1')
    reference_image = File(exists=True, desc='set a new reference image to change the target coordinate system.')
    interpolation = traits.Enum('cubic', 'linear', 'nearest', usedefault=True, argstr='FinalBSplineInterpolationOrder', desc='set a new interpolator for transformation')
    output_type = traits.Enum('float', 'unsigned char', 'unsigned short', 'short', 'unsigned long', 'long', 'double', argstr='ResultImagePixelType', desc='set a new output pixel type for resampled images')
    output_format = traits.Enum('nii.gz', 'nii', 'mhd', 'hdr', 'vtk', argstr='ResultImageFormat', desc='set a new image format for resampled images')
    output_file = File(desc='the filename for the resulting transform file')