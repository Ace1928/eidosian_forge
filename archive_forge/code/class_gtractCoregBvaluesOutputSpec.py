import os
from ...base import (
class gtractCoregBvaluesOutputSpec(TraitedSpec):
    outputVolume = File(desc='Required: name of output NRRD file containing moving images individually resampled and fit to the specified fixed image index.', exists=True)
    outputTransform = File(desc='Registration 3D transforms concatenated in a single output file.  There are no tools that can use this, but can be used for debugging purposes.', exists=True)