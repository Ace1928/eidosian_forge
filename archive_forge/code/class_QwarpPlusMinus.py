import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class QwarpPlusMinus(Qwarp):
    """A version of 3dQwarp for performing field susceptibility correction
    using two images with opposing phase encoding directions.

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> qwarp = afni.QwarpPlusMinus()
    >>> qwarp.inputs.in_file = 'sub-01_dir-LR_epi.nii.gz'
    >>> qwarp.inputs.nopadWARP = True
    >>> qwarp.inputs.base_file = 'sub-01_dir-RL_epi.nii.gz'
    >>> qwarp.cmdline
    '3dQwarp -prefix Qwarp.nii.gz -plusminus -base sub-01_dir-RL_epi.nii.gz -source sub-01_dir-LR_epi.nii.gz -nopadWARP'
    >>> res = warp.run()  # doctest: +SKIP

    See Also
    --------
    For complete details, see the `3dQwarp Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dQwarp.html>`__

    """
    input_spec = QwarpPlusMinusInputSpec