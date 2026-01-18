from nipype.interfaces.base import (
import os
class GradientAnisotropicDiffusionOutputSpec(TraitedSpec):
    outputVolume = File(position=-1, desc='Output filtered', exists=True)