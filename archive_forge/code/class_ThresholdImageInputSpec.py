import os
from warnings import warn
from ..base import traits, isdefined, TraitedSpec, File, Str, InputMultiObject
from ..mixins import CopyHeaderInterface
from .base import ANTSCommandInputSpec, ANTSCommand
class ThresholdImageInputSpec(ANTSCommandInputSpec):
    dimension = traits.Int(3, usedefault=True, position=1, argstr='%d', desc='dimension of output image')
    input_image = File(exists=True, mandatory=True, position=2, argstr='%s', desc='input image file')
    output_image = File(position=3, argstr='%s', name_source=['input_image'], name_template='%s_resampled', desc='output image file', keep_extension=True)
    mode = traits.Enum('Otsu', 'Kmeans', argstr='%s', position=4, requires=['num_thresholds'], xor=['th_low', 'th_high'], desc='whether to run Otsu / Kmeans thresholding')
    num_thresholds = traits.Int(position=5, argstr='%d', desc='number of thresholds')
    input_mask = File(exists=True, requires=['num_thresholds'], argstr='%s', desc='input mask for Otsu, Kmeans')
    th_low = traits.Float(position=4, argstr='%f', xor=['mode'], desc='lower threshold')
    th_high = traits.Float(position=5, argstr='%f', xor=['mode'], desc='upper threshold')
    inside_value = traits.Float(1, position=6, argstr='%f', requires=['th_low'], desc='inside value')
    outside_value = traits.Float(0, position=7, argstr='%f', requires=['th_low'], desc='outside value')
    copy_header = traits.Bool(True, mandatory=True, usedefault=True, desc='copy headers of the original image into the output (corrected) file')