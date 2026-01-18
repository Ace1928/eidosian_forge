from nipype.interfaces.base import (
import os
class IntensityDifferenceMetricOutputSpec(TraitedSpec):
    outputVolume = File(position=-1, desc='Output volume to keep the results of change quantification.', exists=True)
    reportFileName = File(desc='Report file name', exists=True)