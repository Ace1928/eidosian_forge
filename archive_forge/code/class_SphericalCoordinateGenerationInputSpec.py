from nipype.interfaces.base import (
import os
class SphericalCoordinateGenerationInputSpec(CommandLineInputSpec):
    inputAtlasImage = File(desc='Input atlas image', exists=True, argstr='--inputAtlasImage %s')
    outputPath = traits.Str(desc='Output path for rho, phi and theta images', argstr='--outputPath %s')