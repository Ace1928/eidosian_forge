from nipype.interfaces.base import (
import os
class ImageLabelCombine(SEMLikeCommandLine):
    """title: Image Label Combine

    category: Filtering

    description: Combine two label maps into one

    version: 0.1.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/ImageLabelCombine

    contributor: Alex Yarmarkovich (SPL, BWH)
    """
    input_spec = ImageLabelCombineInputSpec
    output_spec = ImageLabelCombineOutputSpec
    _cmd = 'ImageLabelCombine '
    _outputs_filenames = {'OutputLabelMap': 'OutputLabelMap.nii'}