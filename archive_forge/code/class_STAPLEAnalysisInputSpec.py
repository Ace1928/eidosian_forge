import os
from ...base import (
class STAPLEAnalysisInputSpec(CommandLineInputSpec):
    inputDimension = traits.Int(desc='Required: input image Dimension 2 or 3', argstr='--inputDimension %d')
    inputLabelVolume = InputMultiPath(File(exists=True), desc='Required: input label volume', argstr='--inputLabelVolume %s...')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Required: output image', argstr='--outputVolume %s')