import os.path as op
from ..base import (
from .base import MRTrix3Base, MRTrix3BaseInputSpec
class DWIBiasCorrectInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s', position=-2, mandatory=True, desc='input DWI image')
    in_mask = File(argstr='-mask %s', desc='input mask image for bias field estimation')
    use_ants = traits.Bool(argstr='ants', mandatory=True, desc='use ANTS N4 to estimate the inhomogeneity field', position=0, xor=['use_fsl'])
    use_fsl = traits.Bool(argstr='fsl', mandatory=True, desc='use FSL FAST to estimate the inhomogeneity field', position=0, xor=['use_ants'])
    bias = File(argstr='-bias %s', desc='bias field')
    out_file = File(name_template='%s_biascorr', name_source='in_file', keep_extension=True, argstr='%s', position=-1, desc='the output bias corrected DWI image', genfile=True)