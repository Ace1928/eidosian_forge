import os
from ...base import (
class TextureMeasureFilterInputSpec(CommandLineInputSpec):
    inputVolume = File(exists=True, argstr='--inputVolume %s')
    inputMaskVolume = File(exists=True, argstr='--inputMaskVolume %s')
    distance = traits.Int(argstr='--distance %d')
    insideROIValue = traits.Float(argstr='--insideROIValue %f')
    outputFilename = traits.Either(traits.Bool, File(), hash_files=False, argstr='--outputFilename %s')