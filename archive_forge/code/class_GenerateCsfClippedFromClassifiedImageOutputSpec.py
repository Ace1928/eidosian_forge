import os
from ..base import (
class GenerateCsfClippedFromClassifiedImageOutputSpec(TraitedSpec):
    outputVolume = File(desc='Required: output image', exists=True)