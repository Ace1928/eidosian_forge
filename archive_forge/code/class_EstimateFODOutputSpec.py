import os.path as op
from ..base import traits, TraitedSpec, File, InputMultiObject, isdefined
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class EstimateFODOutputSpec(TraitedSpec):
    wm_odf = File(argstr='%s', desc='output WM ODF')
    gm_odf = File(argstr='%s', desc='output GM ODF')
    csf_odf = File(argstr='%s', desc='output CSF ODF')
    predicted_signal = File(desc='output predicted signal')