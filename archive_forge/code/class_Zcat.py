import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class Zcat(AFNICommand):
    """Copies an image of one type to an image of the same
    or different type using 3dZcat command

    For complete details, see the `3dZcat Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dZcat.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> zcat = afni.Zcat()
    >>> zcat.inputs.in_files = ['functional2.nii', 'functional3.nii']
    >>> zcat.inputs.out_file = 'cat_functional.nii'
    >>> zcat.cmdline
    '3dZcat -prefix cat_functional.nii functional2.nii functional3.nii'
    >>> res = zcat.run()  # doctest: +SKIP

    """
    _cmd = '3dZcat'
    input_spec = ZcatInputSpec
    output_spec = AFNICommandOutputSpec