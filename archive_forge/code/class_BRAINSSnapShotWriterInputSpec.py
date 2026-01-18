import os
from ...base import (
class BRAINSSnapShotWriterInputSpec(CommandLineInputSpec):
    inputVolumes = InputMultiPath(File(exists=True), desc='Input image volume list to be extracted as 2D image. Multiple input is possible. At least one input is required.', argstr='--inputVolumes %s...')
    inputBinaryVolumes = InputMultiPath(File(exists=True), desc='Input mask (binary) volume list to be extracted as 2D image. Multiple input is possible.', argstr='--inputBinaryVolumes %s...')
    inputSliceToExtractInPhysicalPoint = InputMultiPath(traits.Float, desc='2D slice number of input images. For autoWorkUp output, which AC-PC aligned, 0,0,0 will be the center.', sep=',', argstr='--inputSliceToExtractInPhysicalPoint %s')
    inputSliceToExtractInIndex = InputMultiPath(traits.Int, desc='2D slice number of input images. For size of 256*256*256 image, 128 is usually used.', sep=',', argstr='--inputSliceToExtractInIndex %s')
    inputSliceToExtractInPercent = InputMultiPath(traits.Int, desc='2D slice number of input images. Percentage input from 0%-100%. (ex. --inputSliceToExtractInPercent 50,50,50', sep=',', argstr='--inputSliceToExtractInPercent %s')
    inputPlaneDirection = InputMultiPath(traits.Int, desc='Plane to display. In general, 0=sagittal, 1=coronal, and 2=axial plane.', sep=',', argstr='--inputPlaneDirection %s')
    outputFilename = traits.Either(traits.Bool, File(), hash_files=False, desc='2D file name of input images. Required.', argstr='--outputFilename %s')