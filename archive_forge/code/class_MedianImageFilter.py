from nipype.interfaces.base import (
import os
class MedianImageFilter(SEMLikeCommandLine):
    """title: Median Image Filter

    category: Filtering.Denoising

    description: The MedianImageFilter is commonly used as a robust approach for noise reduction. This filter is particularly efficient against "salt-and-pepper" noise. In other words, it is robust to the presence of gray-level outliers. MedianImageFilter computes the value of each output pixel as the statistical median of the neighborhood of values around the corresponding input pixel.

    version: 0.1.0.$Revision: 19608 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/MedianImageFilter

    contributor: Bill Lorensen (GE)

    acknowledgements: This command module was derived from Insight/Examples/Filtering/MedianImageFilter (copyright) Insight Software Consortium
    """
    input_spec = MedianImageFilterInputSpec
    output_spec = MedianImageFilterOutputSpec
    _cmd = 'MedianImageFilter '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}