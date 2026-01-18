import os
from ...base import (
class extractNrrdVectorIndexOutputSpec(TraitedSpec):
    outputVolume = File(desc='Required: name of output NRRD file containing the vector image at the given index', exists=True)