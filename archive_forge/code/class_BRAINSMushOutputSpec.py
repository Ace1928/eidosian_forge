import os
from ...base import (
class BRAINSMushOutputSpec(TraitedSpec):
    outputWeightsFile = File(desc='Output Weights File', exists=True)
    outputVolume = File(desc='The MUSH image produced from the T1 and T2 weighted images', exists=True)
    outputMask = File(desc='The brain volume mask generated from the MUSH image', exists=True)