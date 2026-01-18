import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class DWI2TensorInputSpec(CommandLineInputSpec):
    in_file = InputMultiPath(File(exists=True), argstr='%s', mandatory=True, position=-2, desc='Diffusion-weighted images')
    out_filename = File(name_template='%s_tensor.mif', name_source='in_file', output_name='tensor', argstr='%s', desc='Output tensor filename', position=-1)
    encoding_file = File(argstr='-grad %s', position=2, desc='Encoding file supplied as a 4xN text file with each line is in the format [ X Y Z b ], where [ X Y Z ] describe the direction of the applied gradient, and b gives the b-value in units (1000 s/mm^2). See FSL2MRTrix()')
    ignore_slice_by_volume = traits.List(traits.Int, argstr='-ignoreslices %s', sep=' ', position=2, minlen=2, maxlen=2, desc='Requires two values (i.e. [34 1] for [Slice Volume] Ignores the image slices specified when computing the tensor. Slice here means the z coordinate of the slice to be ignored.')
    ignore_volumes = traits.List(traits.Int, argstr='-ignorevolumes %s', sep=' ', position=2, minlen=1, desc='Requires two values (i.e. [2 5 6] for [Volumes] Ignores the image volumes specified when computing the tensor.')
    mask = File(exists=True, argstr='-mask %s', desc='Only perform computation within the specified binary brain mask image.')
    quiet = traits.Bool(argstr='-quiet', position=1, desc='Do not display information messages or progress status.')
    debug = traits.Bool(argstr='-debug', position=1, desc='Display debugging messages.')