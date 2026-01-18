from nipype.interfaces.base import (
import os
class DiffusionTensorScalarMeasurementsOutputSpec(TraitedSpec):
    outputScalar = File(position=-1, desc='Scalar volume derived from tensor', exists=True)