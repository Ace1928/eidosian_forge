import os
from ...base import (
class maxcurvatureInputSpec(CommandLineInputSpec):
    image = File(desc='FA Image', exists=True, argstr='--image %s')
    output = traits.Either(traits.Bool, File(), hash_files=False, desc='Output File', argstr='--output %s')
    sigma = traits.Float(desc='Scale of Gradients', argstr='--sigma %f')
    verbose = traits.Bool(desc='produce verbose output', argstr='--verbose ')