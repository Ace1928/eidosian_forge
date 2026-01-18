from nipype.interfaces.base import (
import os
class AddScalarVolumes(SEMLikeCommandLine):
    """title: Add Scalar Volumes

    category: Filtering.Arithmetic

    description: Adds two images. Although all image types are supported on input, only signed types are produced. The two images do not have to have the same dimensions.

    version: 0.1.0.$Revision: 19608 $(alpha)

    documentation-url: http://slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/Add

    contributor: Bill Lorensen (GE)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    """
    input_spec = AddScalarVolumesInputSpec
    output_spec = AddScalarVolumesOutputSpec
    _cmd = 'AddScalarVolumes '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}