from nipype.interfaces.base import (
import os
class GaussianBlurImageFilterOutputSpec(TraitedSpec):
    outputVolume = File(position=-1, desc='Blurred Volume', exists=True)