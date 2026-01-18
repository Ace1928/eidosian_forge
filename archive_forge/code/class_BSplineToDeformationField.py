from nipype.interfaces.base import (
import os
class BSplineToDeformationField(SEMLikeCommandLine):
    """title: BSpline to deformation field

    category: Legacy.Converters

    description: Create a dense deformation field from a bspline+bulk transform.

    version: 0.1.0.$Revision: 2104 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/BSplineToDeformationField

    contributor: Andrey Fedorov (SPL, BWH)

    acknowledgements: This work is funded by NIH grants R01 CA111288 and U01 CA151261.
    """
    input_spec = BSplineToDeformationFieldInputSpec
    output_spec = BSplineToDeformationFieldOutputSpec
    _cmd = 'BSplineToDeformationField '
    _outputs_filenames = {'defImage': 'defImage.nii'}