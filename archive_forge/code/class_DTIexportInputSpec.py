from nipype.interfaces.base import (
import os
class DTIexportInputSpec(CommandLineInputSpec):
    inputTensor = File(position=-2, desc='Input DTI volume', exists=True, argstr='%s')
    outputFile = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Output DTI file', argstr='%s')