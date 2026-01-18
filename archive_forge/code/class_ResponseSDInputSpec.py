import os.path as op
from ..base import (
from .base import MRTrix3Base, MRTrix3BaseInputSpec
class ResponseSDInputSpec(MRTrix3BaseInputSpec):
    algorithm = traits.Enum('msmt_5tt', 'dhollander', 'tournier', 'tax', argstr='%s', position=1, mandatory=True, desc='response estimation algorithm (multi-tissue)')
    in_file = File(exists=True, argstr='%s', position=-5, mandatory=True, desc='input DWI image')
    mtt_file = File(argstr='%s', position=-4, desc='input 5tt image')
    wm_file = File('wm.txt', argstr='%s', position=-3, usedefault=True, desc='output WM response text file')
    gm_file = File(argstr='%s', position=-2, desc='output GM response text file')
    csf_file = File(argstr='%s', position=-1, desc='output CSF response text file')
    in_mask = File(exists=True, argstr='-mask %s', desc='provide initial mask image')
    max_sh = InputMultiObject(traits.Int, argstr='-lmax %s', sep=',', desc='maximum harmonic degree of response function - single value for single-shell response, list for multi-shell response')