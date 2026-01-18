import os
from ...base import (
class gtractInvertBSplineTransformOutputSpec(TraitedSpec):
    outputTransform = File(desc='Required: output transform file name', exists=True)