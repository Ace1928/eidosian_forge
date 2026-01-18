import os
from ...base import (
class gtractCopyImageOrientationOutputSpec(TraitedSpec):
    outputVolume = File(desc='Required: name of output NRRD or Nifti file containing the reoriented image in reference image space.', exists=True)