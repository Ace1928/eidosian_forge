import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class MRConvertInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-2, desc='input image')
    out_file = File('dwi.mif', argstr='%s', mandatory=True, position=-1, usedefault=True, desc='output image')
    coord = traits.List(traits.Int, sep=' ', argstr='-coord %s', desc='extract data at the specified coordinates')
    vox = traits.List(traits.Float, sep=',', argstr='-vox %s', desc='change the voxel dimensions')
    axes = traits.List(traits.Int, sep=',', argstr='-axes %s', desc='specify the axes that will be used')
    scaling = traits.List(traits.Float, sep=',', argstr='-scaling %s', desc='specify the data scaling parameter')
    json_import = File(exists=True, argstr='-json_import %s', mandatory=False, desc='import data from a JSON file into header key-value pairs')
    json_export = File(exists=False, argstr='-json_export %s', mandatory=False, desc='export data from an image header key-value pairs into a JSON file')