from nipype.interfaces.base import (
import os
class HistogramMatchingInputSpec(CommandLineInputSpec):
    numberOfHistogramLevels = traits.Int(desc='The number of hisogram levels to use', argstr='--numberOfHistogramLevels %d')
    numberOfMatchPoints = traits.Int(desc='The number of match points to use', argstr='--numberOfMatchPoints %d')
    threshold = traits.Bool(desc='If on, only pixels above the mean in each volume are thresholded.', argstr='--threshold ')
    inputVolume = File(position=-3, desc='Input volume to be filtered', exists=True, argstr='%s')
    referenceVolume = File(position=-2, desc='Input volume whose histogram will be matched', exists=True, argstr='%s')
    outputVolume = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Output volume. This is the input volume with intensities matched to the reference volume.', argstr='%s')