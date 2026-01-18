from nipype.interfaces.base import (
import os
class ResampleScalarVolumeOutputSpec(TraitedSpec):
    OutputVolume = File(position=-1, desc='Resampled Volume', exists=True)