import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class TNormInputSpec(AFNICommandInputSpec):
    in_file = File(desc='input file to 3dTNorm', argstr='%s', position=-1, mandatory=True, exists=True, copyfile=False)
    out_file = File(name_template='%s_tnorm', desc='output image file name', argstr='-prefix %s', name_source='in_file')
    norm2 = traits.Bool(desc='L2 normalize (sum of squares = 1) [DEFAULT]', argstr='-norm2')
    normR = traits.Bool(desc='normalize so sum of squares = number of time points \\* e.g., so RMS = 1.', argstr='-normR')
    norm1 = traits.Bool(desc='L1 normalize (sum of absolute values = 1)', argstr='-norm1')
    normx = traits.Bool(desc='Scale so max absolute value = 1 (L_infinity norm)', argstr='-normx')
    polort = traits.Int(desc="Detrend with polynomials of order p before normalizing [DEFAULT = don't do this].\nUse '-polort 0' to remove the mean, for example", argstr='-polort %s')
    L1fit = traits.Bool(desc='Detrend with L1 regression (L2 is the default)\nThis option is here just for the hell of it', argstr='-L1fit')