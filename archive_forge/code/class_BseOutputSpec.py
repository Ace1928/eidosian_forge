import os
import re as regex
from ..base import (
class BseOutputSpec(TraitedSpec):
    outputMRIVolume = File(desc='path/name of brain-masked MRI volume')
    outputMaskFile = File(desc='path/name of smooth brain mask')
    outputDiffusionFilter = File(desc='path/name of diffusion filter output')
    outputEdgeMap = File(desc='path/name of edge map output')
    outputDetailedBrainMask = File(desc='path/name of detailed brain mask')
    outputCortexFile = File(desc='path/name of cortex file')