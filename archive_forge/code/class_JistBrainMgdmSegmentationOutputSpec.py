import os
from ..base import (
class JistBrainMgdmSegmentationOutputSpec(TraitedSpec):
    outSegmented = File(desc='Segmented Brain Image', exists=True)
    outLevelset = File(desc='Levelset Boundary Image', exists=True)
    outPosterior2 = File(desc='Posterior Maximum Memberships (4D)', exists=True)
    outPosterior3 = File(desc='Posterior Maximum Labels (4D)', exists=True)