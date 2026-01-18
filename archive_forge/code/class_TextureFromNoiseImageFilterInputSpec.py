import os
from ...base import (
class TextureFromNoiseImageFilterInputSpec(CommandLineInputSpec):
    inputVolume = File(desc='Required: input image', exists=True, argstr='--inputVolume %s')
    inputRadius = traits.Int(desc='Required: input neighborhood radius', argstr='--inputRadius %d')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Required: output image', argstr='--outputVolume %s')