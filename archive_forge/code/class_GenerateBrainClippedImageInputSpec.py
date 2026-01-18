import os
from ...base import (
class GenerateBrainClippedImageInputSpec(CommandLineInputSpec):
    inputImg = File(desc='input volume 1, usually t1 image', exists=True, argstr='--inputImg %s')
    inputMsk = File(desc='input volume 2, usually t2 image', exists=True, argstr='--inputMsk %s')
    outputFileName = traits.Either(traits.Bool, File(), hash_files=False, desc='(required) output file name', argstr='--outputFileName %s')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')