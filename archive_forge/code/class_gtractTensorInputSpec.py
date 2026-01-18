import os
from ...base import (
class gtractTensorInputSpec(CommandLineInputSpec):
    inputVolume = File(desc='Required: input image 4D NRRD image. Must contain data based on at least 6 distinct diffusion directions. The inputVolume is allowed to have multiple b0 and gradient direction images. Averaging of the b0 image is done internally in this step. Prior averaging of the DWIs is not required.', exists=True, argstr='--inputVolume %s')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Required: name of output NRRD file containing the Tensor vector image', argstr='--outputVolume %s')
    medianFilterSize = InputMultiPath(traits.Int, desc='Median filter radius in all 3 directions', sep=',', argstr='--medianFilterSize %s')
    maskProcessingMode = traits.Enum('NOMASK', 'ROIAUTO', 'ROI', desc='ROIAUTO:  mask is implicitly defined using a otsu foreground and hole filling algorithm. ROI: Uses the masks to define what parts of the image should be used for computing the transform. NOMASK: no mask used', argstr='--maskProcessingMode %s')
    maskVolume = File(desc='Mask Image, if maskProcessingMode is ROI', exists=True, argstr='--maskVolume %s')
    backgroundSuppressingThreshold = traits.Int(desc='Image threshold to suppress background. This sets a threshold used on the b0 image to remove background voxels from processing. Typically, values of 100 and 500 work well for Siemens and GE DTI data, respectively. Check your data particularly in the globus pallidus to make sure the brain tissue is not being eliminated with this threshold.', argstr='--backgroundSuppressingThreshold %d')
    resampleIsotropic = traits.Bool(desc='Flag to resample to isotropic voxels. Enabling this feature is recommended if fiber tracking will be performed.', argstr='--resampleIsotropic ')
    size = traits.Float(desc='Isotropic voxel size to resample to', argstr='--size %f')
    b0Index = traits.Int(desc='Index in input vector index to extract', argstr='--b0Index %d')
    applyMeasurementFrame = traits.Bool(desc='Flag to apply the measurement frame to the gradient directions', argstr='--applyMeasurementFrame ')
    ignoreIndex = InputMultiPath(traits.Int, desc='Ignore diffusion gradient index. Used to remove specific gradient directions with artifacts.', sep=',', argstr='--ignoreIndex %s')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')