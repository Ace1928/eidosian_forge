import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class SurfaceSmoothInputSpec(FSTraitedSpec):
    in_file = File(mandatory=True, argstr='--sval %s', desc='source surface file')
    subject_id = traits.String(mandatory=True, argstr='--s %s', desc='subject id of surface file')
    hemi = traits.Enum('lh', 'rh', argstr='--hemi %s', mandatory=True, desc='hemisphere to operate on')
    fwhm = traits.Float(argstr='--fwhm %.4f', xor=['smooth_iters'], desc='effective FWHM of the smoothing process')
    smooth_iters = traits.Int(argstr='--smooth %d', xor=['fwhm'], desc='iterations of the smoothing process')
    cortex = traits.Bool(True, argstr='--cortex', usedefault=True, desc='only smooth within ``$hemi.cortex.label``')
    reshape = traits.Bool(argstr='--reshape', desc='reshape surface vector to fit in non-mgh format')
    out_file = File(argstr='--tval %s', genfile=True, desc='surface file to write')