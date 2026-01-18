import os
from ...base import (
class CleanUpOverlapLabelsOutputSpec(TraitedSpec):
    outputBinaryVolumes = OutputMultiPath(File(exists=True), desc='The output label map images, with integer values in it. Each label value specified in the inputLabels is combined into this output label map volume')