import os
from ...base import (
class GenerateLabelMapFromProbabilityMapInputSpec(CommandLineInputSpec):
    inputVolumes = InputMultiPath(File(exists=True), desc='The Input probaiblity images to be computed for label maps', argstr='--inputVolumes %s...')
    outputLabelVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='The Input binary image for region of interest', argstr='--outputLabelVolume %s')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')