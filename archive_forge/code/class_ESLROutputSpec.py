import os
from ...base import (
class ESLROutputSpec(TraitedSpec):
    outputVolume = File(desc='Output Label Volume', exists=True)