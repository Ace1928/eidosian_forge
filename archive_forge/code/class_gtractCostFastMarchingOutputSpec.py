import os
from ...base import (
class gtractCostFastMarchingOutputSpec(TraitedSpec):
    outputCostVolume = File(desc='Output vcl_cost image', exists=True)
    outputSpeedVolume = File(desc='Output speed image', exists=True)