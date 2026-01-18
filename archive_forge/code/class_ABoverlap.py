import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class ABoverlap(AFNICommand):
    """Output (to screen) is a count of various things about how
    the automasks of datasets A and B overlap or don't overlap.

    For complete details, see the `3dABoverlap Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dABoverlap.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> aboverlap = afni.ABoverlap()
    >>> aboverlap.inputs.in_file_a = 'functional.nii'
    >>> aboverlap.inputs.in_file_b = 'structural.nii'
    >>> aboverlap.inputs.out_file =  'out.mask_ae_overlap.txt'
    >>> aboverlap.cmdline
    '3dABoverlap functional.nii structural.nii  |& tee out.mask_ae_overlap.txt'
    >>> res = aboverlap.run()  # doctest: +SKIP

    """
    _cmd = '3dABoverlap'
    input_spec = ABoverlapInputSpec
    output_spec = AFNICommandOutputSpec