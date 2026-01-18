import os
import numpy as np
import nibabel as nb
import warnings
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import traits, TraitedSpec, InputMultiPath, File, isdefined
from .base import FSLCommand, FSLCommandInputSpec, Info
class ApplyTOPUPInputSpec(FSLCommandInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True, desc='name of file with images', argstr='--imain=%s', sep=',')
    encoding_file = File(exists=True, mandatory=True, desc='name of text file with PE directions/times', argstr='--datain=%s')
    in_index = traits.List(traits.Int, argstr='--inindex=%s', sep=',', desc='comma separated list of indices corresponding to --datain')
    in_topup_fieldcoef = File(exists=True, argstr='--topup=%s', copyfile=False, requires=['in_topup_movpar'], desc='topup file containing the field coefficients')
    in_topup_movpar = File(exists=True, requires=['in_topup_fieldcoef'], copyfile=False, desc='topup movpar.txt file')
    out_corrected = File(desc='output (warped) image', name_source=['in_files'], name_template='%s_corrected', argstr='--out=%s')
    method = traits.Enum('jac', 'lsr', argstr='--method=%s', desc='use jacobian modulation (jac) or least-squares resampling (lsr)')
    interp = traits.Enum('trilinear', 'spline', argstr='--interp=%s', desc='interpolation method')
    datatype = traits.Enum('char', 'short', 'int', 'float', 'double', argstr='-d=%s', desc='force output data type')