import os
from ...base import (
class gtractTensorOutputSpec(TraitedSpec):
    outputVolume = File(desc='Required: name of output NRRD file containing the Tensor vector image', exists=True)