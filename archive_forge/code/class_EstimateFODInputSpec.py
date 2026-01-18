import os.path as op
from ..base import traits, TraitedSpec, File, InputMultiObject, isdefined
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class EstimateFODInputSpec(MRTrix3BaseInputSpec):
    algorithm = traits.Enum('csd', 'msmt_csd', argstr='%s', position=-8, mandatory=True, desc='FOD algorithm')
    in_file = File(exists=True, argstr='%s', position=-7, mandatory=True, desc='input DWI image')
    wm_txt = File(argstr='%s', position=-6, mandatory=True, desc='WM response text file')
    wm_odf = File('wm.mif', argstr='%s', position=-5, usedefault=True, mandatory=True, desc='output WM ODF')
    gm_txt = File(argstr='%s', position=-4, desc='GM response text file')
    gm_odf = File('gm.mif', usedefault=True, argstr='%s', position=-3, desc='output GM ODF')
    csf_txt = File(argstr='%s', position=-2, desc='CSF response text file')
    csf_odf = File('csf.mif', usedefault=True, argstr='%s', position=-1, desc='output CSF ODF')
    mask_file = File(exists=True, argstr='-mask %s', desc='mask image')
    shell = traits.List(traits.Float, sep=',', argstr='-shell %s', desc='specify one or more dw gradient shells')
    max_sh = InputMultiObject(traits.Int, value=[8], usedefault=True, argstr='-lmax %s', sep=',', desc='maximum harmonic degree of response function - single value for single-shell response, list for multi-shell response')
    in_dirs = File(exists=True, argstr='-directions %s', desc='specify the directions over which to apply the non-negativity constraint (by default, the built-in 300 direction set is used). These should be supplied as a text file containing the [ az el ] pairs for the directions.')
    predicted_signal = File(argstr='-predicted_signal %s', desc="specify a file to contain the predicted signal from the FOD estimates. This can be used to calculate the residual signal.Note that this is only valid if algorithm == 'msmt_csd'. For single shell reconstructions use a combination of SHConv and SH2Amp instead.")