from nipype.interfaces.base import (
import os
class ModelToLabelMap(SEMLikeCommandLine):
    """title: Model To Label Map

    category: Surface Models

    description: Intersects an input model with an reference volume and produces an output label map.

    version: 0.1.0.$Revision: 8643 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/PolyDataToLabelMap

    contributor: Nicole Aucoin (SPL, BWH), Xiaodong Tao (GE)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    """
    input_spec = ModelToLabelMapInputSpec
    output_spec = ModelToLabelMapOutputSpec
    _cmd = 'ModelToLabelMap '
    _outputs_filenames = {'OutputVolume': 'OutputVolume.nii'}