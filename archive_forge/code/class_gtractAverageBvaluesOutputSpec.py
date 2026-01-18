import os
from ...base import (
class gtractAverageBvaluesOutputSpec(TraitedSpec):
    outputVolume = File(desc='Required: name of output NRRD file containing directly averaged baseline images', exists=True)