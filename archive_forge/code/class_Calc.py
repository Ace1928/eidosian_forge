import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class Calc(CommandLine):
    """Compute an expression using MINC files as input.

    Examples
    --------

    >>> from nipype.interfaces.minc import Calc
    >>> from nipype.interfaces.minc.testdata import nonempty_minc_data

    >>> file0 = nonempty_minc_data(0)
    >>> file1 = nonempty_minc_data(1)
    >>> calc = Calc(input_files=[file0, file1], output_file='/tmp/calc.mnc', expression='A[0] + A[1]') # add files together
    >>> calc.run() # doctest: +SKIP
    """
    input_spec = CalcInputSpec
    output_spec = CalcOutputSpec
    _cmd = 'minccalc'