import os
from ...base import (
class gtractResampleB0OutputSpec(TraitedSpec):
    outputVolume = File(desc='Required: name of output NRRD file containing the resampled input image.', exists=True)