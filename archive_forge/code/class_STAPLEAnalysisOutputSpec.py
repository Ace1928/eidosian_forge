import os
from ...base import (
class STAPLEAnalysisOutputSpec(TraitedSpec):
    outputVolume = File(desc='Required: output image', exists=True)