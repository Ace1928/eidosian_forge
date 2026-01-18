import os
from ...base import (
class GradientAnisotropicDiffusionImageFilterOutputSpec(TraitedSpec):
    outputVolume = File(desc='Required: output image', exists=True)