from nipype.interfaces.base import (
import os
class DicomToNrrdConverterInputSpec(CommandLineInputSpec):
    inputDicomDirectory = Directory(desc='Directory holding Dicom series', exists=True, argstr='--inputDicomDirectory %s')
    outputDirectory = traits.Either(traits.Bool, Directory(), hash_files=False, desc='Directory holding the output NRRD format', argstr='--outputDirectory %s')
    outputVolume = traits.Str(desc='Output filename (.nhdr or .nrrd)', argstr='--outputVolume %s')
    smallGradientThreshold = traits.Float(desc='If a gradient magnitude is greater than 0 and less than smallGradientThreshold, then DicomToNrrdConverter will display an error message and quit, unless the useBMatrixGradientDirections option is set.', argstr='--smallGradientThreshold %f')
    writeProtocolGradientsFile = traits.Bool(desc="Write the protocol gradients to a file suffixed by '.txt' as they were specified in the procol by multiplying each diffusion gradient direction by the measurement frame.  This file is for debugging purposes only, the format is not fixed, and will likely change as debugging of new dicom formats is necessary.", argstr='--writeProtocolGradientsFile ')
    useIdentityMeaseurementFrame = traits.Bool(desc='Adjust all the gradients so that the measurement frame is an identity matrix.', argstr='--useIdentityMeaseurementFrame ')
    useBMatrixGradientDirections = traits.Bool(desc='Fill the nhdr header with the gradient directions and bvalues computed out of the BMatrix. Only changes behavior for Siemens data.', argstr='--useBMatrixGradientDirections ')