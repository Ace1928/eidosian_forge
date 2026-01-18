import os
from ..base import (
class JistLaminarVolumetricLayeringOutputSpec(TraitedSpec):
    outContinuous = File(desc='Continuous depth measurement', exists=True)
    outDiscrete = File(desc='Discrete sampled layers', exists=True)
    outLayer = File(desc='Layer boundary surfaces', exists=True)