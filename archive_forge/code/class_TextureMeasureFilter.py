import os
from ...base import (
class TextureMeasureFilter(SEMLikeCommandLine):
    """title: Canny Level Set Image Filter

    category: Filtering.FeatureDetection

    description: The CannySegmentationLevelSet is commonly used to refine a manually generated manual mask.

    version: 0.3.0

    license: CC

    contributor: Regina Kim

    acknowledgements: This command module was derived from Insight/Examples/Segmentation/CannySegmentationLevelSetImageFilter.cxx (copyright) Insight Software Consortium.  See http://wiki.na-mic.org/Wiki/index.php/Slicer3:Execution_Model_Documentation for more detailed descriptions.
    """
    input_spec = TextureMeasureFilterInputSpec
    output_spec = TextureMeasureFilterOutputSpec
    _cmd = ' TextureMeasureFilter '
    _outputs_filenames = {'outputFilename': 'outputFilename'}
    _redirect_x = False