import os
from ...base import (
class insertMidACPCpointOutputSpec(TraitedSpec):
    outputLandmarkFile = File(desc='Output landmark file (.fcsv)', exists=True)