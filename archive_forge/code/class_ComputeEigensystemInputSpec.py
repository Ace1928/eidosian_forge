import os
from ...utils.filemanip import split_filename
from ..base import (
class ComputeEigensystemInputSpec(StdOutCommandLineInputSpec):
    in_file = File(exists=True, argstr='< %s', mandatory=True, position=1, desc='Tensor-fitted data filename')
    inputmodel = traits.Enum('dt', 'multitensor', argstr='-inputmodel %s', desc='Specifies the model that the input data contains parameters for')
    maxcomponents = traits.Int(argstr='-maxcomponents %d', desc='The maximum number of tensor components in a voxel of the input data.')
    inputdatatype = traits.Enum('double', 'float', 'long', 'int', 'short', 'char', argstr='-inputdatatype %s', usedefault=True, desc='Specifies the data type of the input data. The data type can be any of the following strings: "char", "short", "int", "long", "float" or "double".Default is double data type')
    outputdatatype = traits.Enum('double', 'float', 'long', 'int', 'short', 'char', argstr='-outputdatatype %s', usedefault=True, desc='Specifies the data type of the output data.')