import os
from ...base import (
class BRAINSAlignMSPOutputSpec(TraitedSpec):
    OutputresampleMSP = File(desc=',         The image to be output.,       ', exists=True)
    resultsDir = Directory(desc=',         The directory for the results to be written.,       ', exists=True)