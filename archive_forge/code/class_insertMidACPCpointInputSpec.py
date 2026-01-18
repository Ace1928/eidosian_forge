import os
from ...base import (
class insertMidACPCpointInputSpec(CommandLineInputSpec):
    inputLandmarkFile = File(desc='Input landmark file (.fcsv)', exists=True, argstr='--inputLandmarkFile %s')
    outputLandmarkFile = traits.Either(traits.Bool, File(), hash_files=False, desc='Output landmark file (.fcsv)', argstr='--outputLandmarkFile %s')