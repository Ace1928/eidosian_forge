import os
from ...base import (
class BRAINSMultiSTAPLEOutputSpec(TraitedSpec):
    outputMultiSTAPLE = File(desc='the MultiSTAPLE average of input label volumes', exists=True)
    outputConfusionMatrix = File(desc='Confusion Matrix', exists=True)