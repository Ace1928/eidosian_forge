import os
import numpy as np
from ..base import TraitedSpec, File, traits, InputMultiPath, isdefined
from .base import FSLCommand, FSLCommandInputSpec
class MultiImageMathsInput(MathsInput):
    op_string = traits.String(position=4, argstr='%s', mandatory=True, desc='python formatted string of operations to perform')
    operand_files = InputMultiPath(File(exists=True), mandatory=True, desc='list of file names to plug into op string')