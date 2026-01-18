import os
from ...base import (
class gtractClipAnisotropyInputSpec(CommandLineInputSpec):
    inputVolume = File(desc='Required: input image file name', exists=True, argstr='--inputVolume %s')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Required: name of output NRRD file containing the clipped anisotropy image', argstr='--outputVolume %s')
    clipFirstSlice = traits.Bool(desc='Clip the first slice of the anisotropy image', argstr='--clipFirstSlice ')
    clipLastSlice = traits.Bool(desc='Clip the last slice of the anisotropy image', argstr='--clipLastSlice ')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')