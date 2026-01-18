import os
from ...base import (
class BRAINSClipInferiorOutputSpec(TraitedSpec):
    outputVolume = File(desc='Output image, a short int copy of the upper portion of the input image, filled with BackgroundFillValue.', exists=True)