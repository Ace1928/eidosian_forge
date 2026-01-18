import os
from ...base import (
class BRAINSMultiSTAPLE(SEMLikeCommandLine):
    """title: Create best representative label map)

    category: Segmentation.Specialized

    description: given a list of label map images, create a representative/average label map.
    """
    input_spec = BRAINSMultiSTAPLEInputSpec
    output_spec = BRAINSMultiSTAPLEOutputSpec
    _cmd = ' BRAINSMultiSTAPLE '
    _outputs_filenames = {'outputMultiSTAPLE': 'outputMultiSTAPLE.nii', 'outputConfusionMatrix': 'outputConfusionMatrixh5|mat|txt'}
    _redirect_x = False