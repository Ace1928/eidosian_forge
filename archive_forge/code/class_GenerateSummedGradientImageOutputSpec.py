import os
from ...base import (
class GenerateSummedGradientImageOutputSpec(TraitedSpec):
    outputFileName = File(desc='(required) output file name', exists=True)