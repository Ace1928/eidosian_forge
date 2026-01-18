import os
from ...base import (
class ImageRegionPlotterInputSpec(CommandLineInputSpec):
    inputVolume1 = File(desc='The Input image to be computed for statistics', exists=True, argstr='--inputVolume1 %s')
    inputVolume2 = File(desc='The Input image to be computed for statistics', exists=True, argstr='--inputVolume2 %s')
    inputBinaryROIVolume = File(desc='The Input binary image for region of interest', exists=True, argstr='--inputBinaryROIVolume %s')
    inputLabelVolume = File(desc='The Label Image', exists=True, argstr='--inputLabelVolume %s')
    numberOfHistogramBins = traits.Int(desc=' the number of histogram levels', argstr='--numberOfHistogramBins %d')
    outputJointHistogramData = traits.Str(desc=' output data file name', argstr='--outputJointHistogramData %s')
    useROIAUTO = traits.Bool(desc=' Use ROIAUTO to compute region of interest. This cannot be used with inputLabelVolume', argstr='--useROIAUTO ')
    useIntensityForHistogram = traits.Bool(desc=' Create Intensity Joint Histogram instead of Quantile Joint Histogram', argstr='--useIntensityForHistogram ')
    verbose = traits.Bool(desc=' print debugging information,       ', argstr='--verbose ')