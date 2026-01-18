import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class BlurInMaskInputSpec(AFNICommandInputSpec):
    in_file = File(desc='input file to 3dSkullStrip', argstr='-input %s', position=1, mandatory=True, exists=True, copyfile=False)
    out_file = File(name_template='%s_blur', desc='output to the file', argstr='-prefix %s', name_source='in_file', position=-1)
    mask = File(desc='Mask dataset, if desired.  Blurring will occur only within the mask. Voxels NOT in the mask will be set to zero in the output.', argstr='-mask %s')
    multimask = File(desc='Multi-mask dataset -- each distinct nonzero value in dataset will be treated as a separate mask for blurring purposes.', argstr='-Mmask %s')
    automask = traits.Bool(desc='Create an automask from the input dataset.', argstr='-automask')
    fwhm = traits.Float(desc='fwhm kernel size', argstr='-FWHM %f', mandatory=True)
    preserve = traits.Bool(desc='Normally, voxels not in the mask will be set to zero in the output. If you want the original values in the dataset to be preserved in the output, use this option.', argstr='-preserve')
    float_out = traits.Bool(desc='Save dataset as floats, no matter what the input data type is.', argstr='-float')
    options = Str(desc='options', argstr='%s', position=2)