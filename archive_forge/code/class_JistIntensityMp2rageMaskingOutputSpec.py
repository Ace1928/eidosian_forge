import os
from ..base import (
class JistIntensityMp2rageMaskingOutputSpec(TraitedSpec):
    outSignal = File(desc='Signal Proba Image', exists=True)
    outSignal2 = File(desc='Signal Mask Image', exists=True)
    outMasked = File(desc='Masked T1 Map Image', exists=True)
    outMasked2 = File(desc='Masked Iso Image', exists=True)