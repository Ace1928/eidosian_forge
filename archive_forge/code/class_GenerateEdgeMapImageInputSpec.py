import os
from ...base import (
class GenerateEdgeMapImageInputSpec(CommandLineInputSpec):
    inputMRVolumes = InputMultiPath(File(exists=True), desc='List of input structural MR volumes to create the maximum edgemap', argstr='--inputMRVolumes %s...')
    inputMask = File(desc='Input mask file name. If set, image histogram percentiles will be calculated within the mask', exists=True, argstr='--inputMask %s')
    minimumOutputRange = traits.Int(desc='Map lower quantile and below to minimum output range. It should be a small number greater than zero. Default is 1', argstr='--minimumOutputRange %d')
    maximumOutputRange = traits.Int(desc='Map upper quantile and above to maximum output range. Default is 255 that is the maximum range of unsigned char', argstr='--maximumOutputRange %d')
    lowerPercentileMatching = traits.Float(desc='Map lower quantile and below to minOutputRange. It should be a value between zero and one', argstr='--lowerPercentileMatching %f')
    upperPercentileMatching = traits.Float(desc='Map upper quantile and above to maxOutputRange. It should be a value between zero and one', argstr='--upperPercentileMatching %f')
    outputEdgeMap = traits.Either(traits.Bool, File(), hash_files=False, desc='output edgemap file name', argstr='--outputEdgeMap %s')
    outputMaximumGradientImage = traits.Either(traits.Bool, File(), hash_files=False, desc='output gradient image file name', argstr='--outputMaximumGradientImage %s')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')