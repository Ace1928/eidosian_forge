import os
from ...base import (
class DistanceMapsOutputSpec(TraitedSpec):
    outputVolume = File(desc='Required: output image', exists=True)