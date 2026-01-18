import os
from ..base import (
class JistLaminarProfileSamplingOutputSpec(TraitedSpec):
    outProfilemapped = File(desc='Profile-mapped Intensity Image', exists=True)
    outProfile2 = File(desc='Profile 4D Mask', exists=True)