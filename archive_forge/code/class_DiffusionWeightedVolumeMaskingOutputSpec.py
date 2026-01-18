from nipype.interfaces.base import (
import os
class DiffusionWeightedVolumeMaskingOutputSpec(TraitedSpec):
    outputBaseline = File(position=-2, desc='Estimated baseline volume', exists=True)
    thresholdMask = File(position=-1, desc='Otsu Threshold Mask', exists=True)