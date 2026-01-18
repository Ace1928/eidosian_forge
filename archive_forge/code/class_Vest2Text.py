import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class Vest2Text(FSLCommand):
    """
    Use FSL Vest2Text`https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/GLM(2f)CreatingDesignMatricesByHand.html`_
    to convert your design.mat design.con and design.fts files into plain text.

    Examples
    --------
    >>> from nipype.interfaces.fsl import Vest2Text
    >>> v2t = Vest2Text()
    >>> v2t.inputs.in_file = "design.mat"
    >>> v2t.cmdline
    'Vest2Text design.mat design.txt'
    >>> res = v2t.run() # doctest: +SKIP
    """
    input_spec = Vest2TextInputSpec
    output_spec = Vest2TextOutputSpec
    _cmd = 'Vest2Text'