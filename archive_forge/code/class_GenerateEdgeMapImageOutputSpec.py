import os
from ...base import (
class GenerateEdgeMapImageOutputSpec(TraitedSpec):
    outputEdgeMap = File(desc='(required) output file name', exists=True)
    outputMaximumGradientImage = File(desc='output gradient image file name', exists=True)