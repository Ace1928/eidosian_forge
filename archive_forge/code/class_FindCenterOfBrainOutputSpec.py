import os
from ...base import (
class FindCenterOfBrainOutputSpec(TraitedSpec):
    clippedImageMask = File(exists=True)
    debugDistanceImage = File(exists=True)
    debugGridImage = File(exists=True)
    debugAfterGridComputationsForegroundImage = File(exists=True)
    debugClippedImageMask = File(exists=True)
    debugTrimmedImage = File(exists=True)