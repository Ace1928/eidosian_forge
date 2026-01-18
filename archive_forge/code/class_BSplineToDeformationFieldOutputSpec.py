from nipype.interfaces.base import (
import os
class BSplineToDeformationFieldOutputSpec(TraitedSpec):
    defImage = File(exists=True)