from nipype.interfaces.base import (
import os
class FiducialRegistrationOutputSpec(TraitedSpec):
    saveTransform = File(desc='Save the transform that results from registration', exists=True)