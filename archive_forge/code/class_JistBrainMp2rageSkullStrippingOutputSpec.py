import os
from ..base import (
class JistBrainMp2rageSkullStrippingOutputSpec(TraitedSpec):
    outBrain = File(desc='Brain Mask Image', exists=True)
    outMasked = File(desc='Masked T1 Map Image', exists=True)
    outMasked2 = File(desc='Masked T1-weighted Image', exists=True)
    outMasked3 = File(desc='Masked Filter Image', exists=True)