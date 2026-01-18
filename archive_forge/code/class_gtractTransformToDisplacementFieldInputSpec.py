import os
from ...base import (
class gtractTransformToDisplacementFieldInputSpec(CommandLineInputSpec):
    inputTransform = File(desc='Input Transform File Name', exists=True, argstr='--inputTransform %s')
    inputReferenceVolume = File(desc='Required: input image file name to exemplify the anatomical space over which to vcl_express the transform as a displacement field.', exists=True, argstr='--inputReferenceVolume %s')
    outputDeformationFieldVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Output deformation field', argstr='--outputDeformationFieldVolume %s')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')