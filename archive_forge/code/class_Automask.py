import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class Automask(AFNICommand):
    """Create a brain-only mask of the image using AFNI 3dAutomask command

    For complete details, see the `3dAutomask Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dAutomask.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> automask = afni.Automask()
    >>> automask.inputs.in_file = 'functional.nii'
    >>> automask.inputs.dilate = 1
    >>> automask.inputs.outputtype = 'NIFTI'
    >>> automask.cmdline  # doctest: +ELLIPSIS
    '3dAutomask -apply_prefix functional_masked.nii -dilate 1 -prefix functional_mask.nii functional.nii'
    >>> res = automask.run()  # doctest: +SKIP

    """
    _cmd = '3dAutomask'
    input_spec = AutomaskInputSpec
    output_spec = AutomaskOutputSpec