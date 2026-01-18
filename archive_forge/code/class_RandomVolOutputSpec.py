import os
from ..base import (
class RandomVolOutputSpec(TraitedSpec):
    outRand1 = File(desc='Rand1', exists=True)