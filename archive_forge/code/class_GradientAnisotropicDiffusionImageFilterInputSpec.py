import os
from ...base import (
class GradientAnisotropicDiffusionImageFilterInputSpec(CommandLineInputSpec):
    inputVolume = File(desc='Required: input image', exists=True, argstr='--inputVolume %s')
    numberOfIterations = traits.Int(desc='Optional value for number of Iterations', argstr='--numberOfIterations %d')
    timeStep = traits.Float(desc='Time step for diffusion process', argstr='--timeStep %f')
    conductance = traits.Float(desc='Conductance for diffusion process', argstr='--conductance %f')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Required: output image', argstr='--outputVolume %s')