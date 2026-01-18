import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class Maskave(AFNICommand):
    """Computes average of all voxels in the input dataset
    which satisfy the criterion in the options list

    For complete details, see the `3dmaskave Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dmaskave.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> maskave = afni.Maskave()
    >>> maskave.inputs.in_file = 'functional.nii'
    >>> maskave.inputs.mask= 'seed_mask.nii'
    >>> maskave.inputs.quiet= True
    >>> maskave.cmdline  # doctest: +ELLIPSIS
    '3dmaskave -mask seed_mask.nii -quiet functional.nii > functional_maskave.1D'
    >>> res = maskave.run()  # doctest: +SKIP

    """
    _cmd = '3dmaskave'
    input_spec = MaskaveInputSpec
    output_spec = AFNICommandOutputSpec