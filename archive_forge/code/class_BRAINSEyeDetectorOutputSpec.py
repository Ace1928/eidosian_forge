import os
from ...base import (
class BRAINSEyeDetectorOutputSpec(TraitedSpec):
    outputVolume = File(desc='The output volume', exists=True)