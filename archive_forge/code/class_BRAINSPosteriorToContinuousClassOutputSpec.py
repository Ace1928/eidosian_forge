import os
from ...base import (
class BRAINSPosteriorToContinuousClassOutputSpec(TraitedSpec):
    outputVolume = File(desc='Output Continuous Tissue Classified Image', exists=True)