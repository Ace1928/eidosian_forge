import os
from ...base import (
class JointHistogramInputSpec(CommandLineInputSpec):
    inputVolumeInXAxis = File(desc='The Input image to be computed for statistics', exists=True, argstr='--inputVolumeInXAxis %s')
    inputVolumeInYAxis = File(desc='The Input image to be computed for statistics', exists=True, argstr='--inputVolumeInYAxis %s')
    inputMaskVolumeInXAxis = File(desc='Input mask volume for inputVolumeInXAxis. Histogram will be computed just for the masked region', exists=True, argstr='--inputMaskVolumeInXAxis %s')
    inputMaskVolumeInYAxis = File(desc='Input mask volume for inputVolumeInYAxis. Histogram will be computed just for the masked region', exists=True, argstr='--inputMaskVolumeInYAxis %s')
    outputJointHistogramImage = traits.Str(desc=' output joint histogram image file name. Histogram is usually 2D image. ', argstr='--outputJointHistogramImage %s')
    verbose = traits.Bool(desc=' print debugging information,       ', argstr='--verbose ')