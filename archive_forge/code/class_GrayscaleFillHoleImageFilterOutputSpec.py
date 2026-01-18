from nipype.interfaces.base import (
import os
class GrayscaleFillHoleImageFilterOutputSpec(TraitedSpec):
    outputVolume = File(position=-1, desc='Output filtered', exists=True)