import os
import numpy as np
import nibabel as nb
import warnings
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import traits, TraitedSpec, InputMultiPath, File, isdefined
from .base import FSLCommand, FSLCommandInputSpec, Info
class PrepareFieldmapInputSpec(FSLCommandInputSpec):
    scanner = traits.String('SIEMENS', argstr='%s', position=1, desc='must be SIEMENS', usedefault=True)
    in_phase = File(exists=True, argstr='%s', position=2, mandatory=True, desc='Phase difference map, in SIEMENS format range from 0-4096 or 0-8192)')
    in_magnitude = File(exists=True, argstr='%s', position=3, mandatory=True, desc='Magnitude difference map, brain extracted')
    delta_TE = traits.Float(2.46, usedefault=True, mandatory=True, argstr='%f', position=-2, desc='echo time difference of the fieldmap sequence in ms. (usually 2.46ms in Siemens)')
    nocheck = traits.Bool(False, position=-1, argstr='--nocheck', usedefault=True, desc='do not perform sanity checks for image size/range/dimensions')
    out_fieldmap = File(argstr='%s', position=4, desc='output name for prepared fieldmap')