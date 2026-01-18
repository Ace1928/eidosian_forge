from nipype.interfaces.base import (
import os
class GrayscaleModelMakerOutputSpec(TraitedSpec):
    OutputGeometry = File(position=-1, desc='Output that contains geometry model.', exists=True)