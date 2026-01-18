import os
from ...base import (
class GeneratePurePlugMaskInputSpec(CommandLineInputSpec):
    inputImageModalities = InputMultiPath(File(exists=True), desc='List of input image file names to create pure plugs mask', argstr='--inputImageModalities %s...')
    threshold = traits.Float(desc='threshold value to define class membership', argstr='--threshold %f')
    numberOfSubSamples = InputMultiPath(traits.Int, desc='Number of continuous index samples taken at each direction of lattice space for each plug volume', sep=',', argstr='--numberOfSubSamples %s')
    outputMaskFile = traits.Either(traits.Bool, File(), hash_files=False, desc='Output binary mask file name', argstr='--outputMaskFile %s')