import os.path as op
from ..base import traits, TraitedSpec, File, InputMultiObject, isdefined
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class ConstrainedSphericalDeconvolutionInputSpec(EstimateFODInputSpec):
    gm_odf = File(argstr='%s', position=-3, desc='output GM ODF')
    csf_odf = File(argstr='%s', position=-1, desc='output CSF ODF')
    max_sh = InputMultiObject(traits.Int, argstr='-lmax %s', sep=',', desc='maximum harmonic degree of response function - single value for single-shell response, list for multi-shell response')