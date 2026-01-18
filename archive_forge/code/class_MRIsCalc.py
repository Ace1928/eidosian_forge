import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class MRIsCalc(FSCommand):
    """
    'mris_calc' is a simple calculator that operates on FreeSurfer
    curvatures and volumes. In most cases, the calculator functions with
    three arguments: two inputs and an <ACTION> linking them. Some
    actions, however, operate with only one input <file1>. In all cases,
    the first input <file1> is the name of a FreeSurfer curvature overlay
    (e.g. rh.curv) or volume file (e.g. orig.mgz). For two inputs, the
    calculator first assumes that the second input is a file. If, however,
    this second input file doesn't exist, the calculator assumes it refers
    to a float number, which is then processed according to <ACTION>.Note:
    <file1> and <file2> should typically be generated on the same subject.

    Examples
    ========
    >>> from nipype.interfaces.freesurfer import MRIsCalc
    >>> example = MRIsCalc()
    >>> example.inputs.in_file1 = 'lh.area' # doctest: +SKIP
    >>> example.inputs.in_file2 = 'lh.area.pial' # doctest: +SKIP
    >>> example.inputs.action = 'add'
    >>> example.inputs.out_file = 'area.mid'
    >>> example.cmdline # doctest: +SKIP
    'mris_calc -o lh.area.mid lh.area add lh.area.pial'
    """
    _cmd = 'mris_calc'
    input_spec = MRIsCalcInputSpec
    output_spec = MRIsCalcOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs