from nipype.interfaces.base import (
import os
class MedianImageFilterOutputSpec(TraitedSpec):
    outputVolume = File(position=-1, desc='Output filtered', exists=True)