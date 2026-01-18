import os
from ...base import (
class UnbiasedNonLocalMeansOutputSpec(TraitedSpec):
    outputVolume = File(position=-1, desc='Output (filtered) MRI volume.', exists=True)