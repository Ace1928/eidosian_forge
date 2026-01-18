from nipype.interfaces.base import (
import os
class EMSegmentTransformToNewFormat(SEMLikeCommandLine):
    """title:
      Transform MRML Files to New EMSegmenter Standard


    category:
      Utilities


    description:
      Transform MRML Files to New EMSegmenter Standard

    """
    input_spec = EMSegmentTransformToNewFormatInputSpec
    output_spec = EMSegmentTransformToNewFormatOutputSpec
    _cmd = 'EMSegmentTransformToNewFormat '
    _outputs_filenames = {'outputMRMLFileName': 'outputMRMLFileName.mrml'}