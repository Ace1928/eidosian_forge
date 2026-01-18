import os
from ...base import (
class ShuffleVectorsModuleOutputSpec(TraitedSpec):
    outputVectorFileBaseName = File(desc='output vector file name prefix. Usually end with .txt and header file has prost fix of .txt.hdr', exists=True)