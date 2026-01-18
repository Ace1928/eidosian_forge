import os.path as op
from ..base import (
from .base import MRTrix3Base, MRTrix3BaseInputSpec
class ReplaceFSwithFIRSTInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-4, desc='input anatomical image')
    in_t1w = File(exists=True, argstr='%s', mandatory=True, position=-3, desc='input T1 image')
    in_config = File(exists=True, argstr='%s', position=-2, desc='connectome configuration file')
    out_file = File('aparc+first.mif', argstr='%s', mandatory=True, position=-1, usedefault=True, desc='output file after processing')