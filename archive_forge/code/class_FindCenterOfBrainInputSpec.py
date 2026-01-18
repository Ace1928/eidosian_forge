import os
from ...base import (
class FindCenterOfBrainInputSpec(CommandLineInputSpec):
    inputVolume = File(desc='The image in which to find the center.', exists=True, argstr='--inputVolume %s')
    imageMask = File(exists=True, argstr='--imageMask %s')
    clippedImageMask = traits.Either(traits.Bool, File(), hash_files=False, argstr='--clippedImageMask %s')
    maximize = traits.Bool(argstr='--maximize ')
    axis = traits.Int(argstr='--axis %d')
    otsuPercentileThreshold = traits.Float(argstr='--otsuPercentileThreshold %f')
    closingSize = traits.Int(argstr='--closingSize %d')
    headSizeLimit = traits.Float(argstr='--headSizeLimit %f')
    headSizeEstimate = traits.Float(argstr='--headSizeEstimate %f')
    backgroundValue = traits.Int(argstr='--backgroundValue %d')
    generateDebugImages = traits.Bool(argstr='--generateDebugImages ')
    debugDistanceImage = traits.Either(traits.Bool, File(), hash_files=False, argstr='--debugDistanceImage %s')
    debugGridImage = traits.Either(traits.Bool, File(), hash_files=False, argstr='--debugGridImage %s')
    debugAfterGridComputationsForegroundImage = traits.Either(traits.Bool, File(), hash_files=False, argstr='--debugAfterGridComputationsForegroundImage %s')
    debugClippedImageMask = traits.Either(traits.Bool, File(), hash_files=False, argstr='--debugClippedImageMask %s')
    debugTrimmedImage = traits.Either(traits.Bool, File(), hash_files=False, argstr='--debugTrimmedImage %s')