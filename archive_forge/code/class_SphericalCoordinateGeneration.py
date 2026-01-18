from nipype.interfaces.base import (
import os
class SphericalCoordinateGeneration(SEMLikeCommandLine):
    """title: Spherical Coordinate Generation

    category: Testing.FeatureDetection

    description: get the atlas image as input and generates the rho, phi and theta images.

    version: 0.1.0.$Revision: 1 $(alpha)

    contributor: Ali Ghayoor
    """
    input_spec = SphericalCoordinateGenerationInputSpec
    output_spec = SphericalCoordinateGenerationOutputSpec
    _cmd = ' SphericalCoordinateGeneration '
    _outputs_filenames = {}
    _redirect_x = False