import os
from ...base import (
class BRAINSMushInputSpec(CommandLineInputSpec):
    inputFirstVolume = File(desc='Input image (1) for mixture optimization', exists=True, argstr='--inputFirstVolume %s')
    inputSecondVolume = File(desc='Input image (2) for mixture optimization', exists=True, argstr='--inputSecondVolume %s')
    inputMaskVolume = File(desc='Input label image for mixture optimization', exists=True, argstr='--inputMaskVolume %s')
    outputWeightsFile = traits.Either(traits.Bool, File(), hash_files=False, desc='Output Weights File', argstr='--outputWeightsFile %s')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='The MUSH image produced from the T1 and T2 weighted images', argstr='--outputVolume %s')
    outputMask = traits.Either(traits.Bool, File(), hash_files=False, desc='The brain volume mask generated from the MUSH image', argstr='--outputMask %s')
    seed = InputMultiPath(traits.Int, desc='Seed Point for Brain Region Filling', sep=',', argstr='--seed %s')
    desiredMean = traits.Float(desc='Desired mean within the mask for weighted sum of both images.', argstr='--desiredMean %f')
    desiredVariance = traits.Float(desc='Desired variance within the mask for weighted sum of both images.', argstr='--desiredVariance %f')
    lowerThresholdFactorPre = traits.Float(desc='Lower threshold factor for finding an initial brain mask', argstr='--lowerThresholdFactorPre %f')
    upperThresholdFactorPre = traits.Float(desc='Upper threshold factor for finding an initial brain mask', argstr='--upperThresholdFactorPre %f')
    lowerThresholdFactor = traits.Float(desc='Lower threshold factor for defining the brain mask', argstr='--lowerThresholdFactor %f')
    upperThresholdFactor = traits.Float(desc='Upper threshold factor for defining the brain mask', argstr='--upperThresholdFactor %f')
    boundingBoxSize = InputMultiPath(traits.Int, desc='Size of the cubic bounding box mask used when no brain mask is present', sep=',', argstr='--boundingBoxSize %s')
    boundingBoxStart = InputMultiPath(traits.Int, desc='XYZ point-coordinate for the start of the cubic bounding box mask used when no brain mask is present', sep=',', argstr='--boundingBoxStart %s')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')