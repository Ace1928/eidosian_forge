from nipype.interfaces.base import (
import os
class ThresholdScalarVolume(SEMLikeCommandLine):
    """title: Threshold Scalar Volume

    category: Filtering

    description: <p>Threshold an image.</p><p>Set image values to a user-specified outside value if they are below, above, or between simple threshold values.</p><p>ThresholdAbove: The values greater than or equal to the threshold value are set to OutsideValue.</p><p>ThresholdBelow: The values less than or equal to the threshold value are set to OutsideValue.</p><p>ThresholdOutside: The values outside the range Lower-Upper are set to OutsideValue.</p><p>Although all image types are supported on input, only signed types are produced.</p><p>

    version: 0.1.0.$Revision: 2104 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/Threshold

    contributor: Nicole Aucoin (SPL, BWH), Ron Kikinis (SPL, BWH)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    """
    input_spec = ThresholdScalarVolumeInputSpec
    output_spec = ThresholdScalarVolumeOutputSpec
    _cmd = 'ThresholdScalarVolume '
    _outputs_filenames = {'OutputVolume': 'OutputVolume.nii'}