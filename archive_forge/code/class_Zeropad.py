import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class Zeropad(AFNICommand):
    """Adds planes of zeros to a dataset (i.e., pads it out).

    For complete details, see the `3dZeropad Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dZeropad.html>`__

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> zeropad = afni.Zeropad()
    >>> zeropad.inputs.in_files = 'functional.nii'
    >>> zeropad.inputs.out_file = 'pad_functional.nii'
    >>> zeropad.inputs.I = 10
    >>> zeropad.inputs.S = 10
    >>> zeropad.inputs.A = 10
    >>> zeropad.inputs.P = 10
    >>> zeropad.inputs.R = 10
    >>> zeropad.inputs.L = 10
    >>> zeropad.cmdline
    '3dZeropad -A 10 -I 10 -L 10 -P 10 -R 10 -S 10 -prefix pad_functional.nii functional.nii'
    >>> res = zeropad.run()  # doctest: +SKIP

    """
    _cmd = '3dZeropad'
    input_spec = ZeropadInputSpec
    output_spec = AFNICommandOutputSpec