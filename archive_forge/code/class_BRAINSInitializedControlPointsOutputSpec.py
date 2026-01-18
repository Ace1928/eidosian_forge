import os
from ...base import (
class BRAINSInitializedControlPointsOutputSpec(TraitedSpec):
    outputVolume = File(desc='Output Volume', exists=True)