from nipype.interfaces.base import (
import os
class SimpleRegionGrowingSegmentation(SEMLikeCommandLine):
    """title: Simple Region Growing Segmentation

    category: Segmentation

    description: A simple region growing segmentation algorithm based on intensity statistics. To create a list of fiducials (Seeds) for this algorithm, click on the tool bar icon of an arrow pointing to a starburst fiducial to enter the 'place a new object mode' and then use the fiducials module. This module uses the Slicer Command Line Interface (CLI) and the ITK filters CurvatureFlowImageFilter and ConfidenceConnectedImageFilter.

    version: 0.1.0.$Revision: 19904 $(alpha)

    documentation-url: http://www.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/SimpleRegionGrowingSegmentation

    contributor: Jim Miller (GE)

    acknowledgements: This command module was derived from Insight/Examples (copyright) Insight Software Consortium
    """
    input_spec = SimpleRegionGrowingSegmentationInputSpec
    output_spec = SimpleRegionGrowingSegmentationOutputSpec
    _cmd = 'SimpleRegionGrowingSegmentation '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}