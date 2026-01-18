from nipype.interfaces.base import (
import os
class RobustStatisticsSegmenterInputSpec(CommandLineInputSpec):
    expectedVolume = traits.Float(desc='The approximate volume of the object, in mL.', argstr='--expectedVolume %f')
    intensityHomogeneity = traits.Float(desc='What is the homogeneity of intensity within the object? Given constant intensity at 1.0 score and extreme fluctuating intensity at 0.', argstr='--intensityHomogeneity %f')
    curvatureWeight = traits.Float(desc='Given sphere 1.0 score and extreme rough boundary/surface 0 score, what is the expected smoothness of the object?', argstr='--curvatureWeight %f')
    labelValue = traits.Int(desc='Label value of the output image', argstr='--labelValue %d')
    maxRunningTime = traits.Float(desc='The program will stop if this time is reached.', argstr='--maxRunningTime %f')
    originalImageFileName = File(position=-3, desc='Original image to be segmented', exists=True, argstr='%s')
    labelImageFileName = File(position=-2, desc='Label image for initialization', exists=True, argstr='%s')
    segmentedImageFileName = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Segmented image', argstr='%s')