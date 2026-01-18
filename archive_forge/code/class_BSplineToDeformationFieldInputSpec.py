from nipype.interfaces.base import (
import os
class BSplineToDeformationFieldInputSpec(CommandLineInputSpec):
    tfm = File(exists=True, argstr='--tfm %s')
    refImage = File(exists=True, argstr='--refImage %s')
    defImage = traits.Either(traits.Bool, File(), hash_files=False, argstr='--defImage %s')