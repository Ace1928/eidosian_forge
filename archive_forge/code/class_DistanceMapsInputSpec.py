import os
from ...base import (
class DistanceMapsInputSpec(CommandLineInputSpec):
    inputLabelVolume = File(desc='Required: input tissue label image', exists=True, argstr='--inputLabelVolume %s')
    inputMaskVolume = File(desc='Required: input brain mask image', exists=True, argstr='--inputMaskVolume %s')
    inputTissueLabel = traits.Int(desc='Required: input integer value of tissue type used to calculate distance', argstr='--inputTissueLabel %d')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Required: output image', argstr='--outputVolume %s')