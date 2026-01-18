from nipype.interfaces.base import (
import os
class ModelToLabelMapOutputSpec(TraitedSpec):
    OutputVolume = File(position=-1, desc='The label volume', exists=True)