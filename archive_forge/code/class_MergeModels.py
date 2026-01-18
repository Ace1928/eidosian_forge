from nipype.interfaces.base import (
import os
class MergeModels(SEMLikeCommandLine):
    """title: Merge Models

    category: Surface Models

    description: Merge the polydata from two input models and output a new model with the added polydata. Uses the vtkAppendPolyData filter. Works on .vtp and .vtk surface files.

    version: $Revision$

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/MergeModels

    contributor: Nicole Aucoin (SPL, BWH), Ron Kikinis (SPL, BWH), Daniel Haehn (SPL, BWH)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    """
    input_spec = MergeModelsInputSpec
    output_spec = MergeModelsOutputSpec
    _cmd = 'MergeModels '
    _outputs_filenames = {'ModelOutput': 'ModelOutput.vtk'}