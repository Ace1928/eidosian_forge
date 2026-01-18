from nipype.interfaces.base import (
import os
class ThresholdScalarVolumeOutputSpec(TraitedSpec):
    OutputVolume = File(position=-1, desc='Thresholded input volume', exists=True)