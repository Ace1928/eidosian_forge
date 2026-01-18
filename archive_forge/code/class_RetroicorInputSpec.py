import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class RetroicorInputSpec(AFNICommandInputSpec):
    in_file = File(desc='input file to 3dretroicor', argstr='%s', position=-1, mandatory=True, exists=True, copyfile=False)
    out_file = File(name_template='%s_retroicor', name_source=['in_file'], desc='output image file name', argstr='-prefix %s', position=1)
    card = File(desc='1D cardiac data file for cardiac correction', argstr='-card %s', position=-2, exists=True)
    resp = File(desc='1D respiratory waveform data for correction', argstr='-resp %s', position=-3, exists=True)
    threshold = traits.Int(desc='Threshold for detection of R-wave peaks in input (Make sure it is above the background noise level, Try 3/4 or 4/5 times range plus minimum)', argstr='-threshold %d', position=-4)
    order = traits.Int(desc='The order of the correction (2 is typical)', argstr='-order %s', position=-5)
    cardphase = File(desc='Filename for 1D cardiac phase output', argstr='-cardphase %s', position=-6, hash_files=False)
    respphase = File(desc='Filename for 1D resp phase output', argstr='-respphase %s', position=-7, hash_files=False)