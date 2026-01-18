import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class SUSANInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=1, desc='filename of input timeseries')
    brightness_threshold = traits.Float(argstr='%.10f', position=2, mandatory=True, desc='brightness threshold and should be greater than noise level and less than contrast of edges to be preserved.')
    fwhm = traits.Float(argstr='%.10f', position=3, mandatory=True, desc='fwhm of smoothing, in mm, gets converted using sqrt(8*log(2))')
    dimension = traits.Enum(3, 2, argstr='%d', position=4, usedefault=True, desc='within-plane (2) or fully 3D (3)')
    use_median = traits.Enum(1, 0, argstr='%d', position=5, usedefault=True, desc='whether to use a local median filter in the cases where single-point noise is detected')
    usans = traits.List(traits.Tuple(File(exists=True), traits.Float), maxlen=2, argstr='', position=6, usedefault=True, desc='determines whether the smoothing area (USAN) is to be found from secondary images (0, 1 or 2). A negative value for any brightness threshold will auto-set the threshold at 10% of the robust range')
    out_file = File(argstr='%s', position=-1, genfile=True, desc='output file name', hash_files=False)