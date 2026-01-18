import os
from ...base import (
class gtractConcatDwiOutputSpec(TraitedSpec):
    outputVolume = File(desc='Required: name of output NRRD file containing the combined diffusion weighted images.', exists=True)