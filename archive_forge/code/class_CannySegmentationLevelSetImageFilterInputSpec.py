import os
from ...base import (
class CannySegmentationLevelSetImageFilterInputSpec(CommandLineInputSpec):
    inputVolume = File(exists=True, argstr='--inputVolume %s')
    initialModel = File(exists=True, argstr='--initialModel %s')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, argstr='--outputVolume %s')
    outputSpeedVolume = traits.Either(traits.Bool, File(), hash_files=False, argstr='--outputSpeedVolume %s')
    cannyThreshold = traits.Float(desc='Canny Threshold Value', argstr='--cannyThreshold %f')
    cannyVariance = traits.Float(desc='Canny variance', argstr='--cannyVariance %f')
    advectionWeight = traits.Float(desc='Controls the smoothness of the resulting mask, small number are more smooth, large numbers allow more sharp corners.  ', argstr='--advectionWeight %f')
    initialModelIsovalue = traits.Float(desc="The identification of the input model iso-surface.  (for a binary image with 0s and 1s use 0.5) (for a binary image with 0s and 255's use 127.5).", argstr='--initialModelIsovalue %f')
    maxIterations = traits.Int(desc='The', argstr='--maxIterations %d')