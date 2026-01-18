from nipype.interfaces.base import (
import os
class GrayscaleFillHoleImageFilterInputSpec(CommandLineInputSpec):
    inputVolume = File(position=-2, desc='Input volume to be filtered', exists=True, argstr='%s')
    outputVolume = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Output filtered', argstr='%s')