import os
import numpy as np
from ..base import TraitedSpec, File, traits, InputMultiPath, isdefined
from .base import FSLCommand, FSLCommandInputSpec
class MultiImageMaths(MathsCommand):
    """Use fslmaths to perform a sequence of mathematical operations.

    Examples
    --------
    >>> from nipype.interfaces.fsl import MultiImageMaths
    >>> maths = MultiImageMaths()
    >>> maths.inputs.in_file = "functional.nii"
    >>> maths.inputs.op_string = "-add %s -mul -1 -div %s"
    >>> maths.inputs.operand_files = ["functional2.nii", "functional3.nii"]
    >>> maths.inputs.out_file = "functional4.nii"
    >>> maths.cmdline
    'fslmaths functional.nii -add functional2.nii -mul -1 -div functional3.nii functional4.nii'

    """
    input_spec = MultiImageMathsInput

    def _format_arg(self, name, spec, value):
        if name == 'op_string':
            return value % tuple(self.inputs.operand_files)
        return super(MultiImageMaths, self)._format_arg(name, spec, value)