import os
from ...base import (
class GenerateTestImageInputSpec(CommandLineInputSpec):
    inputVolume = File(desc='input volume 1, usually t1 image', exists=True, argstr='--inputVolume %s')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='(required) output file name', argstr='--outputVolume %s')
    lowerBoundOfOutputVolume = traits.Float(argstr='--lowerBoundOfOutputVolume %f')
    upperBoundOfOutputVolume = traits.Float(argstr='--upperBoundOfOutputVolume %f')
    outputVolumeSize = traits.Float(desc='output Volume Size', argstr='--outputVolumeSize %f')