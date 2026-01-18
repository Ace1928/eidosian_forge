import os
from ...utils.filemanip import split_filename
from ..base import (
class MESDInputSpec(StdOutCommandLineInputSpec):
    in_file = File(exists=True, argstr='-inputfile %s', mandatory=True, position=1, desc='voxel-order data filename')
    inverter = traits.Enum('SPIKE', 'PAS', argstr='-filter %s', position=2, mandatory=True, desc='\nThe inversion index specifies the type of inversion to perform on the data.\nThe currently available choices are:\n\n  +----------------+---------------------------------------------+\n  | Inverter name  | Inverter parameters                         |\n  +================+=============================================+\n  | SPIKE          | bd (b-value x diffusivity along the fibre.) |\n  +----------------+---------------------------------------------+\n  | PAS            | r                                           |\n  +----------------+---------------------------------------------+\n\n')
    inverter_param = traits.Float(argstr='%f', units='NA', position=3, mandatory=True, desc='Parameter associated with the inverter. Cf. inverter description formore information.')
    fastmesd = traits.Bool(argstr='-fastmesd', requires=['mepointset'], desc='Turns off numerical integration checks and fixes the integration point set size at that ofthe index specified by -basepointset..')
    mepointset = traits.Int(argstr='-mepointset %d', units='NA', desc='Use a set of directions other than those in the scheme file for the deconvolution kernel.The number refers to the number of directions on the unit sphere. For example, "-mepointset 54" uses the directions in "camino/PointSets/Elec054.txt".')
    scheme_file = File(exists=True, argstr='-schemefile %s', mandatory=True, desc='Specifies the scheme file for the diffusion MRI data')
    bgmask = File(exists=True, argstr='-bgmask %s', desc='background mask')
    inputdatatype = traits.Enum('float', 'char', 'short', 'int', 'long', 'double', argstr='-inputdatatype %s', desc='Specifies the data type of the input file: "char", "short", "int", "long","float" or "double". The input file must have BIG-ENDIAN ordering.By default, the input type is "float".')