import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class Mesh2PVEInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-3, desc='input mesh')
    reference = File(exists=True, argstr='%s', mandatory=True, position=-2, desc='input reference image')
    in_first = File(exists=True, argstr='-first %s', desc='indicates that the mesh file is provided by FSL FIRST')
    out_file = File('mesh2volume.nii.gz', argstr='%s', mandatory=True, position=-1, usedefault=True, desc='output file containing SH coefficients')