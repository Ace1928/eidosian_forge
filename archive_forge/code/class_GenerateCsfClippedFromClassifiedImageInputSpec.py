import os
from ..base import (
class GenerateCsfClippedFromClassifiedImageInputSpec(CommandLineInputSpec):
    inputCassifiedVolume = File(desc='Required: input tissue label image', exists=True, argstr='--inputCassifiedVolume %s')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Required: output image', argstr='--outputVolume %s')