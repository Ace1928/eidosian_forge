import os
from ...base import (
class TextureFromNoiseImageFilterOutputSpec(TraitedSpec):
    outputVolume = File(desc='Required: output image', exists=True)