from nipype.interfaces.base import (
import os
class ExtractSkeletonOutputSpec(TraitedSpec):
    OutputImageFileName = File(position=-1, desc='Skeleton of the input image', exists=True)