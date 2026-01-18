import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class ComplexInputSpec(FSLCommandInputSpec):
    complex_in_file = File(exists=True, argstr='%s', position=2)
    complex_in_file2 = File(exists=True, argstr='%s', position=3)
    real_in_file = File(exists=True, argstr='%s', position=2)
    imaginary_in_file = File(exists=True, argstr='%s', position=3)
    magnitude_in_file = File(exists=True, argstr='%s', position=2)
    phase_in_file = File(exists=True, argstr='%s', position=3)
    _ofs = ['complex_out_file', 'magnitude_out_file', 'phase_out_file', 'real_out_file', 'imaginary_out_file']
    _conversion = ['real_polar', 'real_cartesian', 'complex_cartesian', 'complex_polar', 'complex_split', 'complex_merge']
    complex_out_file = File(genfile=True, argstr='%s', position=-3, xor=_ofs + _conversion[:2])
    magnitude_out_file = File(genfile=True, argstr='%s', position=-4, xor=_ofs[:1] + _ofs[3:] + _conversion[1:])
    phase_out_file = File(genfile=True, argstr='%s', position=-3, xor=_ofs[:1] + _ofs[3:] + _conversion[1:])
    real_out_file = File(genfile=True, argstr='%s', position=-4, xor=_ofs[:3] + _conversion[:1] + _conversion[2:])
    imaginary_out_file = File(genfile=True, argstr='%s', position=-3, xor=_ofs[:3] + _conversion[:1] + _conversion[2:])
    start_vol = traits.Int(position=-2, argstr='%d')
    end_vol = traits.Int(position=-1, argstr='%d')
    real_polar = traits.Bool(argstr='-realpolar', xor=_conversion, position=1)
    real_cartesian = traits.Bool(argstr='-realcartesian', xor=_conversion, position=1)
    complex_cartesian = traits.Bool(argstr='-complex', xor=_conversion, position=1)
    complex_polar = traits.Bool(argstr='-complexpolar', xor=_conversion, position=1)
    complex_split = traits.Bool(argstr='-complexsplit', xor=_conversion, position=1)
    complex_merge = traits.Bool(argstr='-complexmerge', xor=_conversion + ['start_vol', 'end_vol'], position=1)