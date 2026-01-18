import os
from ...base import (
class gtractResampleDWIInPlaceInputSpec(CommandLineInputSpec):
    inputVolume = File(desc='Required: input image is a 4D NRRD image.', exists=True, argstr='--inputVolume %s')
    referenceVolume = File(desc='If provided, resample to the final space of the referenceVolume 3D data set.', exists=True, argstr='--referenceVolume %s')
    outputResampledB0 = traits.Either(traits.Bool, File(), hash_files=False, desc='Convenience function for extracting the first index location (assumed to be the B0)', argstr='--outputResampledB0 %s')
    inputTransform = File(desc='Required: transform file derived from rigid registration of b0 image to reference structural image.', exists=True, argstr='--inputTransform %s')
    warpDWITransform = File(desc='Optional: transform file to warp gradient volumes.', exists=True, argstr='--warpDWITransform %s')
    debugLevel = traits.Int(desc='Display debug messages, and produce debug intermediate results.  0=OFF, 1=Minimal, 10=Maximum debugging.', argstr='--debugLevel %d')
    imageOutputSize = InputMultiPath(traits.Int, desc='The voxel lattice for the output image, padding is added if necessary. NOTE: if 0,0,0, then the inputVolume size is used.', sep=',', argstr='--imageOutputSize %s')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Required: output image (NRRD file) that has been rigidly transformed into the space of the structural image and padded if image padding was changed from 0,0,0 default.', argstr='--outputVolume %s')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')