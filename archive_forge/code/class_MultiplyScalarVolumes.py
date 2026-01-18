from nipype.interfaces.base import (
import os
class MultiplyScalarVolumes(SEMLikeCommandLine):
    """title: Multiply Scalar Volumes

    category: Filtering.Arithmetic

    description: Multiplies two images. Although all image types are supported on input, only signed types are produced. The two images do not have to have the same dimensions.

    version: 0.1.0.$Revision: 8595 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/Multiply

    contributor: Bill Lorensen (GE)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    """
    input_spec = MultiplyScalarVolumesInputSpec
    output_spec = MultiplyScalarVolumesOutputSpec
    _cmd = 'MultiplyScalarVolumes '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}