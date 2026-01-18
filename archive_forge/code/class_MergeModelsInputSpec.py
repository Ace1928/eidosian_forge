from nipype.interfaces.base import (
import os
class MergeModelsInputSpec(CommandLineInputSpec):
    Model1 = File(position=-3, desc='Model', exists=True, argstr='%s')
    Model2 = File(position=-2, desc='Model', exists=True, argstr='%s')
    ModelOutput = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Model', argstr='%s')