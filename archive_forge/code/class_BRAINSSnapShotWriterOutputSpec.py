import os
from ...base import (
class BRAINSSnapShotWriterOutputSpec(TraitedSpec):
    outputFilename = File(desc='2D file name of input images. Required.', exists=True)