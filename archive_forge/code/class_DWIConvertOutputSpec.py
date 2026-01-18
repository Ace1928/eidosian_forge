import os
from ...base import (
class DWIConvertOutputSpec(TraitedSpec):
    outputVolume = File(desc='Output filename (.nhdr or .nrrd)', exists=True)
    outputBValues = File(desc='The B Values are stored in FSL .bval text file format (defaults to <outputVolume>.bval)', exists=True)
    outputBVectors = File(desc='The Gradient Vectors are stored in FSL .bvec text file format (defaults to <outputVolume>.bvec)', exists=True)
    outputDirectory = Directory(desc='Directory holding the output NRRD file', exists=True)
    gradientVectorFile = File(desc='Text file giving gradient vectors', exists=True)