from nipype.interfaces.base import (
import os
class IntensityDifferenceMetric(SEMLikeCommandLine):
    """title:
      Intensity Difference Change Detection (FAST)


    category:
      Quantification.ChangeQuantification


    description:
      Quantifies the changes between two spatially aligned images based on the pixel-wise difference of image intensities.


    version: 0.1

    contributor: Andrey Fedorov

    acknowledgements:

    """
    input_spec = IntensityDifferenceMetricInputSpec
    output_spec = IntensityDifferenceMetricOutputSpec
    _cmd = 'IntensityDifferenceMetric '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii', 'reportFileName': 'reportFileName'}