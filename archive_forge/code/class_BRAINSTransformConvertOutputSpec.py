import os
from ...base import (
class BRAINSTransformConvertOutputSpec(TraitedSpec):
    displacementVolume = File(exists=True)
    outputTransform = File(exists=True)