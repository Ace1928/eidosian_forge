from nipype.interfaces.base import (
import os
class MergeModelsOutputSpec(TraitedSpec):
    ModelOutput = File(position=-1, desc='Model', exists=True)