from nipype.interfaces.base import (
import os
class GrayscaleModelMaker(SEMLikeCommandLine):
    """title: Grayscale Model Maker

    category: Surface Models

    description: Create 3D surface models from grayscale data. This module uses Marching Cubes to create an isosurface at a given threshold. The resulting surface consists of triangles that separate a volume into regions below and above the threshold. The resulting surface can be smoothed and decimated. This model works on continuous data while the module Model Maker works on labeled (or discrete) data.

    version: 3.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/GrayscaleModelMaker

    license: slicer3

    contributor: Nicole Aucoin (SPL, BWH), Bill Lorensen (GE)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    """
    input_spec = GrayscaleModelMakerInputSpec
    output_spec = GrayscaleModelMakerOutputSpec
    _cmd = 'GrayscaleModelMaker '
    _outputs_filenames = {'OutputGeometry': 'OutputGeometry.vtk'}