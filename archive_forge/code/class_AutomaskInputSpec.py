import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class AutomaskInputSpec(AFNICommandInputSpec):
    in_file = File(desc='input file to 3dAutomask', argstr='%s', position=-1, mandatory=True, exists=True, copyfile=False)
    out_file = File(name_template='%s_mask', desc='output image file name', argstr='-prefix %s', name_source='in_file')
    brain_file = File(name_template='%s_masked', desc='output file from 3dAutomask', argstr='-apply_prefix %s', name_source='in_file')
    clfrac = traits.Float(desc='sets the clip level fraction (must be 0.1-0.9). A small value will tend to make the mask larger [default = 0.5].', argstr='-clfrac %s')
    dilate = traits.Int(desc='dilate the mask outwards', argstr='-dilate %s')
    erode = traits.Int(desc='erode the mask inwards', argstr='-erode %s')