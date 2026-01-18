from nipype.interfaces.base import (
import os
class DiffusionWeightedVolumeMaskingInputSpec(CommandLineInputSpec):
    inputVolume = File(position=-4, desc='Input DWI volume', exists=True, argstr='%s')
    outputBaseline = traits.Either(traits.Bool, File(), position=-2, hash_files=False, desc='Estimated baseline volume', argstr='%s')
    thresholdMask = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Otsu Threshold Mask', argstr='%s')
    otsuomegathreshold = traits.Float(desc='Control the sharpness of the threshold in the Otsu computation. 0: lower threshold, 1: higher threshold', argstr='--otsuomegathreshold %f')
    removeislands = traits.Bool(desc='Remove Islands in Threshold Mask?', argstr='--removeislands ')