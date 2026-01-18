import os
from ...base import (
class NeighborhoodMeanInputSpec(CommandLineInputSpec):
    inputVolume = File(desc='Required: input image', exists=True, argstr='--inputVolume %s')
    inputMaskVolume = File(desc='Required: input brain mask image', exists=True, argstr='--inputMaskVolume %s')
    inputRadius = traits.Int(desc='Required: input neighborhood radius', argstr='--inputRadius %d')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Required: output image', argstr='--outputVolume %s')