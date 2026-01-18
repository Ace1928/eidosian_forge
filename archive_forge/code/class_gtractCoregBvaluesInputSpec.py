import os
from ...base import (
class gtractCoregBvaluesInputSpec(CommandLineInputSpec):
    movingVolume = File(desc='Required: input moving image file name. In order to register gradients within a scan to its first gradient, set the movingVolume and fixedVolume as the same image.', exists=True, argstr='--movingVolume %s')
    fixedVolume = File(desc='Required: input fixed image file name. It is recommended that this image should either contain or be a b0 image.', exists=True, argstr='--fixedVolume %s')
    fixedVolumeIndex = traits.Int(desc='Index in the fixed image for registration. It is recommended that this image should be a b0 image.', argstr='--fixedVolumeIndex %d')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Required: name of output NRRD file containing moving images individually resampled and fit to the specified fixed image index.', argstr='--outputVolume %s')
    outputTransform = traits.Either(traits.Bool, File(), hash_files=False, desc='Registration 3D transforms concatenated in a single output file.  There are no tools that can use this, but can be used for debugging purposes.', argstr='--outputTransform %s')
    eddyCurrentCorrection = traits.Bool(desc='Flag to perform eddy current correction in addition to motion correction (recommended)', argstr='--eddyCurrentCorrection ')
    numberOfIterations = traits.Int(desc='Number of iterations in each 3D fit', argstr='--numberOfIterations %d')
    numberOfSpatialSamples = traits.Int(desc='The number of voxels sampled for mutual information computation.  Increase this for a slower, more careful fit. NOTE that it is suggested to use samplingPercentage instead of this option. However, if set, it overwrites the samplingPercentage option.  ', argstr='--numberOfSpatialSamples %d')
    samplingPercentage = traits.Float(desc='This is a number in (0.0,1.0] interval that shows the percentage of the input fixed image voxels that are sampled for mutual information computation.  Increase this for a slower, more careful fit. You can also limit the sampling focus with ROI masks and ROIAUTO mask generation. The default is to use approximately 5% of voxels (for backwards compatibility 5% ~= 500000/(256*256*256)). Typical values range from 1% for low detail images to 20% for high detail images.', argstr='--samplingPercentage %f')
    relaxationFactor = traits.Float(desc='Fraction of gradient from Jacobian to attempt to move in each 3D fit step (adjust when eddyCurrentCorrection is enabled; suggested value = 0.25)', argstr='--relaxationFactor %f')
    maximumStepSize = traits.Float(desc='Maximum permitted step size to move in each 3D fit step (adjust when eddyCurrentCorrection is enabled; suggested value = 0.1)', argstr='--maximumStepSize %f')
    minimumStepSize = traits.Float(desc='Minimum required step size to move in each 3D fit step without converging -- decrease this to make the fit more exacting', argstr='--minimumStepSize %f')
    spatialScale = traits.Float(desc='How much to scale up changes in position compared to unit rotational changes in radians -- decrease this to put more rotation in the fit', argstr='--spatialScale %f')
    registerB0Only = traits.Bool(desc='Register the B0 images only', argstr='--registerB0Only ')
    debugLevel = traits.Int(desc='Display debug messages, and produce debug intermediate results.  0=OFF, 1=Minimal, 10=Maximum debugging.', argstr='--debugLevel %d')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')