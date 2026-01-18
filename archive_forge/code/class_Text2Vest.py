import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class Text2Vest(FSLCommand):
    """
    Use FSL Text2Vest`https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/GLM(2f)CreatingDesignMatricesByHand.html`_
    to convert your plain text design matrix data into the format used by the FSL tools.

    Examples
    --------
    >>> from nipype.interfaces.fsl import Text2Vest
    >>> t2v = Text2Vest()
    >>> t2v.inputs.in_file = "design.txt"
    >>> t2v.inputs.out_file = "design.mat"
    >>> t2v.cmdline
    'Text2Vest design.txt design.mat'
    >>> res = t2v.run() # doctest: +SKIP
    """
    input_spec = Text2VestInputSpec
    output_spec = Text2VestOutputSpec
    _cmd = 'Text2Vest'