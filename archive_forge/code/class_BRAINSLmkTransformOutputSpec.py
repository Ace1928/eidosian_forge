import os
from ...base import (
class BRAINSLmkTransformOutputSpec(TraitedSpec):
    outputAffineTransform = File(desc='The filename for the estimated affine transform,             ', exists=True)
    outputResampledVolume = File(desc='The filename of the output resampled volume', exists=True)