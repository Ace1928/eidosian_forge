from nipype.interfaces.base import (
import os
class MedianImageFilterInputSpec(CommandLineInputSpec):
    neighborhood = InputMultiPath(traits.Int, desc='The size of the neighborhood in each dimension', sep=',', argstr='--neighborhood %s')
    inputVolume = File(position=-2, desc='Input volume to be filtered', exists=True, argstr='%s')
    outputVolume = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Output filtered', argstr='%s')